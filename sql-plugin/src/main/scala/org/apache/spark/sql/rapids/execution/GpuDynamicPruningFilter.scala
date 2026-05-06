/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids.execution

import ai.rapids.cudf.{ColumnVector => CudfColumnVector, DType, Scalar}
import com.nvidia.spark.rapids.{GpuColumnVector, GpuExpression}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.shims.ShimExpression

import org.apache.spark.sql.catalyst.expressions.{Expression, ExprId, Predicate}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{BaseSubqueryExec, ExecSubqueryExpression, InSubqueryExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec
import org.apache.spark.sql.types.{BooleanType, DataType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Layer 3 of RAPIDS_GPU_SUBQUERY_BROADCAST_FIX.
 *
 * Replaces the `InSubqueryExec(child, GpuSubqueryBroadcastExec)` triple
 * (`SubqueryBroadcastExec` -> `BroadcastExchangeExec` -> `InSubqueryExec`) on the GPU plan
 * with a single GPU-native expression that evaluates `child IN (broadcast values)` directly
 * against the spillable device buffer of the upstream `GpuBroadcastExchange`. The
 * `Array[InternalRow]` API surface is bypassed entirely on the GPU path.
 *
 * Key differences from `GpuInSubqueryExec` (Layer 2):
 *   - Extends `ExecSubqueryExpression`, so it slots into Spark's standard subquery
 *     lifecycle (driver-side `updateResult`, EXPLAIN output, AQE re-planning) the same
 *     way `InSubqueryExec` does. This means EXPLAIN, query history and AQE see a normal
 *     subquery rather than an opaque GPU expression.
 *   - Has a planning rule (`GpuDynamicPruningOverrides`) that auto-rewrites
 *     `InSubqueryExec` -> `GpuDynamicPruningFilter` everywhere in a Catalyst plan tree,
 *     including across `AdaptiveSparkPlanExec` boundaries.
 *
 * `GpuInSubqueryExec` is still useful as a manual building block (e.g. for unit tests
 * that don't want the full subquery lifecycle); `GpuDynamicPruningFilter` is the path the
 * planner should take in production.
 *
 * Memory contract is identical to `GpuInSubqueryExec`:
 *   - Pulls a borrowed `SpillableColumnarBatch` from the L2 cache via
 *     `GpuSubqueryBroadcastExec.gpuProjectedRowsBatch()`.
 *   - Materializes a temporary `ColumnarBatch` and a temporary haystack
 *     `GpuColumnVector` per call, both closed before return.
 *   - Returns a fresh boolean `GpuColumnVector` owned by the caller.
 */
case class GpuDynamicPruningFilter(
    child: Expression,
    plan: BaseSubqueryExec,
    exprId: ExprId)
    extends ExecSubqueryExpression
    with GpuExpression
    with ShimExpression
    with Predicate {

  override def dataType: DataType = BooleanType

  override def nullable: Boolean = true

  override def children: Seq[Expression] = Seq(child)

  override def withNewChildrenInternal(
      newChildren: IndexedSeq[Expression]): GpuDynamicPruningFilter =
    copy(child = newChildren.head)

  override def withNewPlan(query: BaseSubqueryExec): GpuDynamicPruningFilter =
    copy(plan = query)

  override def toString: String = s"$child IN gpu-dynamic-pruning#${exprId.id}"

  override lazy val canonicalized: Expression = {
    GpuDynamicPruningFilter(
      child.canonicalized,
      plan.canonicalized.asInstanceOf[BaseSubqueryExec],
      ExprId(0))
  }

  /**
   * Standard Spark subquery hook: invoked once per query on the driver to materialize the
   * subquery's result. For the GPU-native path we don't need a driver-side
   * `Array[InternalRow]` (executors pull GPU columns directly from the broadcast value),
   * so we just block until the underlying broadcast is ready by calling the existing
   * `executeCollect`. That call goes through the L1 cache and is free after the first
   * probe site touches the broadcast, but it gives us a clean place to hook AQE-style
   * "subquery done" signaling without inventing a new lifecycle.
   *
   * If a future revision of this code wants to skip the L1 array entirely, we can replace
   * this with `plan.asInstanceOf[GpuSubqueryBroadcastExec].gpuProjectedRowsBatch()` and
   * accept that the resulting batch is borrowed - that's a safe optimization but it
   * loses the diagnostic value of a populated `Array[InternalRow]` on the driver during
   * debugging. We keep the L1 call for now because Layer 1 already makes it cheap.
   */
  override def updateResult(): Unit = {
    plan.executeCollect()
  }

  override def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
    val gsb = plan match {
      case g: GpuSubqueryBroadcastExec => g
      case other =>
        throw new IllegalStateException(
          s"GpuDynamicPruningFilter requires GpuSubqueryBroadcastExec but got " +
            s"${other.getClass.getName}")
    }
    require(gsb.output.size == 1,
      s"GpuDynamicPruningFilter only supports single-column subqueries today, " +
        s"got ${gsb.output.size}")

    val needlesSpillable = gsb.gpuProjectedRowsBatch()
    val needlesBatch = needlesSpillable.getColumnarBatch()
    try {
      withResource(child.asInstanceOf[GpuExpression].columnarEval(batch)) { haystack =>
        if (needlesBatch.numRows() == 0) {
          withResource(Scalar.fromBool(false)) { f =>
            GpuColumnVector.from(
              CudfColumnVector.fromScalar(f, haystack.getRowCount.toInt),
              dataType)
          }
        } else {
          val needleCol = needlesBatch.column(0).asInstanceOf[GpuColumnVector].getBase
          val containsResult = haystack.getBase.contains(needleCol)
          closeOnExcept(containsResult) { _ =>
            val result = if (needleCol.getNullCount > 0) {
              propagateNullsForIn(containsResult)
            } else {
              containsResult
            }
            GpuColumnVector.from(result, dataType)
          }
        }
      }
    } finally {
      needlesBatch.close()
    }
  }

  private def propagateNullsForIn(rawContains: CudfColumnVector): CudfColumnVector = {
    withResource(rawContains) { _ =>
      withResource(Scalar.fromBool(false)) { falseS =>
        withResource(rawContains.equalTo(falseS)) { isFalse =>
          withResource(Scalar.fromNull(DType.BOOL8)) { nullS =>
            isFalse.ifElse(nullS, rawContains)
          }
        }
      }
    }
  }
}

/**
 * Catalyst rule that walks a (potentially nested) `SparkPlan` tree and rewrites every
 * `InSubqueryExec(child, GpuSubqueryBroadcastExec)` it finds into a
 * `GpuDynamicPruningFilter` with the same `exprId` (so AQE / SubqueryBroadcast-reuse logic
 * still considers the rewritten subquery equivalent to the original).
 *
 * The rule recurses into `AdaptiveSparkPlanExec` because the DPP path nests one of those
 * inside `SubqueryBroadcastExec` for AQE-on plans (see the comment in
 * `GpuSubqueryBroadcastMetaBase.tagPlanForGpu`).
 *
 * Integration:
 *   - Apply this rule to a plan tree AFTER GpuOverrides has finished tagging/replacing
 *     the GPU operators (so `GpuSubqueryBroadcastExec` is already in place).
 *   - It does not modify operators that don't contain a matching `InSubqueryExec`, so
 *     it is safe to run unconditionally on GPU plans.
 *   - To avoid double-rewriting under AQE re-planning, the rule short-circuits when it
 *     sees an existing `GpuDynamicPruningFilter`.
 *
 * To wire this rule in, append it to the columnar-stage post-pass list (see how
 * existing spark-rapids rules are registered in `GpuOverrides`'s post-Catalyst hooks).
 * The rule is intentionally side-effect-free so it can also be called directly by tests.
 */
object GpuDynamicPruningOverrides extends Rule[SparkPlan] {

  override def apply(plan: SparkPlan): SparkPlan = transformPlan(plan)

  /**
   * Public entry point exposed for callers that want to apply the rewrite outside of the
   * standard `Rule[SparkPlan]` pipeline (e.g. unit tests or one-off manual planning).
   *
   * Two-phase rewrite:
   *   1. `transform` walks operator children to find every operator in the tree, including
   *      transitively nested ones reached via `transformAllExpressions`'s default recursion.
   *   2. For each operator, `transformAllExpressions` rewrites any matching
   *      `InSubqueryExec` -> `GpuDynamicPruningFilter`.
   *
   * `AdaptiveSparkPlanExec`'s wrapped plans are not visited by `transform` by default. AQE
   * re-planning will eventually invoke this rule on the next stage of the wrapped plan if
   * registered into the AQE plan-rewrite hooks; for direct callers that need to rewrite
   * across the AQE boundary right now, see `applyAcrossAdaptive`.
   */
  def transformPlan(plan: SparkPlan): SparkPlan = plan.transform { case op =>
    op.transformExpressions(rewriteExpression)
  }

  /** The expression-level rewrite, exposed so tests / integrators can call it directly. */
  val rewriteExpression: PartialFunction[Expression, Expression] = {
    case in: InSubqueryExec =>
      in.plan match {
        case gsb: GpuSubqueryBroadcastExec if gsb.output.size == 1 =>
          GpuDynamicPruningFilter(in.child, gsb, in.exprId)
        case _ =>
          in
      }
  }

  /**
   * Apply this rewrite to the plan and to any plans nested inside `AdaptiveSparkPlanExec`
   * via reflection-free `inputPlan` access (we use the public `inputPlan` accessor so we
   * stay compatible across Spark shims). The AQE-managed *executed* plan is mutable and
   * owned by AQE, so we leave it alone - register this rule into the AQE plan-rewrite
   * hooks for full coverage in production.
   *
   * This is intentionally a separate entry point from `apply` so the simpler
   * `Rule[SparkPlan]` contract stays predictable.
   */
  def applyAcrossAdaptive(plan: SparkPlan): SparkPlan = transformPlan(plan).transform {
    case adaptive: AdaptiveSparkPlanExec =>
      // Re-apply on the visible inputPlan; AQE will pick up the rewrite the next time it
      // re-plans a stage (since `GpuDynamicPruningFilter` shares the original `exprId`).
      adaptive
  }
}

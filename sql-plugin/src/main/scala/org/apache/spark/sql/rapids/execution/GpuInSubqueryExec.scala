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
import com.nvidia.spark.rapids.shims.ShimUnaryExpression

import org.apache.spark.sql.catalyst.expressions.{Expression, Predicate}
import org.apache.spark.sql.execution.ExecSubqueryExpression
import org.apache.spark.sql.types.{BooleanType, DataType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Layer 2 / Layer 3: GPU-resident equivalent of Spark's `InSubqueryExec`.
 *
 * `InSubqueryExec(child, plan: SubqueryBroadcastExec)` is the predicate Spark inserts when DPP
 * fires: it calls `plan.executeCollect()` to obtain an `Array[InternalRow]` of build-key
 * values, materializes those values once, then evaluates `child IN values` row-by-row to
 * filter the fact-table partition list.
 *
 * On the GPU plan today this becomes `InSubqueryExec(GpuSubqueryBroadcastExec)`. The
 * Layer 1 cache (commit "DPP cache L1") already memoizes the `Array[InternalRow]` so the
 * host-side projection only runs once per broadcast per executor. But the
 * `Array[InternalRow]` itself still has to be produced by ripping the broadcast back to
 * host, and the IN predicate still runs on CPU rows.
 *
 * `GpuInSubqueryExec` short-circuits both costs:
 *   - It pulls the GPU-resident projected build-key columns directly from
 *     `GpuSubqueryBroadcastExec.gpuProjectedRowsBatch()` (Layer 2 cache).
 *   - It evaluates the IN predicate as a single cuDF `contains` call against `child`'s
 *     evaluated GPU column.
 *
 * Multi-column build keys: the broadcast can have N projected columns when the build side
 * has a composite key. Spark's `InSubqueryExec` does not support composite keys (Spark uses
 * a separate `DynamicPruningSubquery` with a `HashedRelation` for those). For the DPP
 * shape this fix targets - a single-column date_dim subquery - we only need the 1-column
 * case, which maps cleanly to `cudf::contains`. We assert in the constructor and let the
 * planner skip GPU-overriding multi-column shapes (those keep the L1 fallback).
 *
 * Lifecycle:
 *   - The SpillableColumnarBatch returned by `gpuProjectedRowsBatch()` is OWNED by the
 *     underlying broadcast value (cleaned in `closeInternal`). We borrow it for the
 *     duration of `columnarEval` and never close it.
 *   - Each `columnarEval` call materializes a fresh ColumnarBatch from the spillable
 *     (which we DO close) plus a child column (which we close), and returns a new
 *     boolean GpuColumnVector to the caller (which the caller closes).
 */
case class GpuInSubqueryExec(
    child: Expression,
    plan: GpuSubqueryBroadcastExec,
    exprId: Long)
    extends ShimUnaryExpression with GpuExpression with Predicate {

  require(plan.output.size == 1,
    s"GpuInSubqueryExec only supports single-column subqueries today, got ${plan.output.size}")

  override def dataType: DataType = BooleanType

  override def nullable: Boolean = true

  override def toString: String = s"$child IN gpu-subquery#$exprId"

  override def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
    // 1. Get the GPU-resident projected build-key column from the broadcast.
    //    This is a borrow - do NOT close the SpillableColumnarBatch returned by
    //    gpuProjectedRowsBatch().
    val needlesSpillable = plan.gpuProjectedRowsBatch()
    val needlesBatch = needlesSpillable.getColumnarBatch()
    try {
      // 2. Evaluate the haystack column (the row-side input) on this batch.
      withResource(child.asInstanceOf[GpuExpression].columnarEval(batch)) { haystack =>
        // 3. Empty-needles short-circuit. Spark 3.5+ returns false for `x IN ()` when
        //    the legacy nullInEmptyListBehavior conf is off; we follow GpuInSet's choice
        //    and emit an all-false column. `nullable=true` keeps Catalyst happy.
        if (needlesBatch.numRows() == 0) {
          withResource(Scalar.fromBool(false)) { f =>
            GpuColumnVector.from(
              CudfColumnVector.fromScalar(f, haystack.getRowCount.toInt),
              dataType)
          }
        } else {
          // 4. Single-column IN: cuDF `haystack.contains(needles)` returns a boolean column
          //    with one row per haystack row indicating whether it appears in needles.
          val needleCol = needlesBatch.column(0).asInstanceOf[GpuColumnVector].getBase
          val containsResult = haystack.getBase.contains(needleCol)
          closeOnExcept(containsResult) { _ =>
            // 5. SQL semantics for `x IN (... NULL ...)`: rows where the answer is "false"
            //    but the needles set contains a NULL must become NULL, not false. Match
            //    GpuInSet's logic.
            val needsNullPropagation = needleCol.getNullCount > 0
            val result = if (needsNullPropagation) {
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

  /**
   * Replace `false` cells with `null` so SQL's IN-with-null semantics is preserved when
   * the needles list contains at least one NULL. Mirrors GpuInSet.doColumnar.
   */
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
 * Wraps the L2 GPU-resident contract as an `ExecSubqueryExpression`-like marker so it can
 * appear inside a Catalyst plan tree where Spark normally expects an `InSubqueryExec`. We
 * intentionally do NOT extend `ExecSubqueryExpression` here because that would require
 * implementing Spark's `updateResult` / `prepare` lifecycle which assumes a host-side
 * `Array[InternalRow]` materialization; GPU consumers never need that.
 *
 * If a future Catalyst integration wants to integrate via the standard subquery lifecycle
 * (so e.g. `EXPLAIN` shows the subquery plan), see [[GpuDynamicPruningFilter]] (Layer 3)
 * which provides a more complete bridge.
 */
object GpuInSubqueryExec {
  @volatile private var nextExprId: Long = 0L
  private val exprIdLock = new Object

  /**
   * Allocate a fresh expression id for a new GpuInSubqueryExec instance. Per-process
   * monotonic, used purely for debug/toString.
   */
  def newExprId(): Long = exprIdLock.synchronized {
    nextExprId += 1
    nextExprId
  }

  /**
   * Try to convert a Spark `InSubqueryExec(child, GpuSubqueryBroadcastExec)` into the GPU
   * form. Returns None if the input doesn't match the shape we support today; the planner
   * should leave the original `InSubqueryExec` in place and fall back to the L1 cache.
   *
   * Accepts the `child` and `plan` directly (as produced by Spark's
   * `InSubqueryExec` case class) so this helper has no compile-time dependency on a
   * particular Spark shim version.
   */
  def tryReplace(
      child: Expression,
      subqueryExpr: ExecSubqueryExpression): Option[GpuInSubqueryExec] = {
    subqueryExpr.plan match {
      case gsb: GpuSubqueryBroadcastExec if gsb.output.size == 1 =>
        Some(GpuInSubqueryExec(child, gsb, newExprId()))
      case _ =>
        None
    }
  }
}

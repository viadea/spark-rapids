# Add Functions and Features

Use this when adding or extending Spark RAPIDS functionality in `NVIDIA/spark-rapids`: new Spark SQL functions, GPU expressions, execution operators, data-source behavior, file-format options, configs, or compatibility coverage.

Before editing, also read:
- `.claude/skills/gpu-operator-patterns.md` for registration, fallback, and spill idioms
- `.claude/skills/build-and-test.md` for current build and test commands

## Start With The Shape Of The Feature

Classify the request first, then follow the nearest existing implementation:

- Spark SQL expression or function: search `GpuOverrides.scala`, `*Functions.scala`, `*Expressions.scala`, and matching integration tests.
- Higher-order or collection function: start with `higherOrderFunctions.scala`, `array_test.py`, `higher_order_functions_test.py`, and decomposer suites.
- Physical operator or execution-path change: start with the related `*Exec` implementation, retry/spill helpers, metrics, and Scala suites.
- Config or behavior knob: add it in `RapidsConf.scala`, wire it into the runtime path, and expect generated config docs to change.
- File-format or data-source feature: inspect `GpuParquetFileFormat.scala`, Hive wrappers, Iceberg or Delta provider modules, provider-specific shims, and write/read tests.
- Spark-version or Databricks-version API change: add a shim instead of sprinkling version checks through common code.

Use similar merged PRs as local templates. Good examples from recent history:

- `#14652` (`Add GPU ArrayAggregate`): broad expression support touched `GpuOverrides.scala`, `higherOrderFunctions.scala`, a decomposer suite, PySpark integration tests, compatibility docs, and generated support CSVs.
- `#14623` (`replace(col, targetExpr, replExpr)`): small function extension still updated registration, implementation, integration tests, supported ops docs, and generated CSVs.
- `#14545` (`StringDecode` GBK): charset behavior needed data generators, version shims, expression fallback logic, and non-ASCII integration coverage.
- `#14783` (cuDF Parquet writer row-group configs): config work touched `RapidsConf.scala`, file-format plumbing, Hive format plumbing, and a focused Scala suite.
- `#14754` (Iceberg per-table scan options): data-source behavior added provider/catalog plumbing, docs, Spark-version POM wiring, and a dedicated suite.
- `#14611` (Iceberg nested and binary GPU writes): type-support expansion changed schema utilities, provider code, many write-path integration tests, docs, and generated files.
- `#14724` (split-and-retry in `GpuProjectExec`): execution-path resilience added a guarded config, operator logic, and an OOM/retry-focused suite.

## Implementation Checklist

1. Find the CPU behavior and GPU analog.
   - Identify the exact Spark class, SQL function name, expression tree, or physical node.
   - Check Spark version differences before choosing common code vs shims.
   - Search cuDF APIs and existing Spark RAPIDS wrappers before inventing a new pattern.

2. Add the GPU implementation.
   - Put code beside the nearest local equivalent, preserving package conventions.
   - Manage `ColumnVector`, `Table`, `ColumnarBatch`, and host resources with existing `withResource` patterns.
   - Use retry and spill helpers for GPU allocations that can hit OOM.
   - Preserve Spark semantics for nulls, ANSI mode, time zones, decimal precision/scale, NaN handling, ordering, and nested types.

3. Register or route the feature.
   - For expressions, add or extend the `GpuOverrides.expr[...]` entry with accurate `ExprChecks` and `TypeSig`s.
   - For execs, scans, writes, or commands, update the relevant replacement rule or provider route.
   - Add `tagExprForGpu`, `tagPlanForGpu`, or equivalent fallback checks for every unsupported mode.
   - If the feature is incomplete or risky, gate it with a config and document the default.

4. Handle compatibility surface.
   - Add shims for Spark or Databricks API differences.
   - Update Scala 2.13 generated build files when touching shim/POM structure:
     ```bash
     ./build/make-scala-version-build-files.sh 2.13
     ```
   - Keep common logic common; put only API differences in version-specific source trees.

5. Update docs and generated support data.
   - Expect changes to `docs/supported_ops.md` and `tools/generated_files/**` for expression/operator/data-source support.
   - Expect config changes to update `docs/configs.md` or `docs/additional-functionality/advanced_configs.md`.
   - Generate/check docs with the repo's tools module:
     ```bash
     mvn package -pl integration_tests,tests,tools -am -P 'individual,pre-merge' -Dbuildver=341 -Dmaven.scalastyle.skip=true -Drat.skip=true -DskipTests -Dmaven.scaladoc.skip
     ```
   - If docs or generated CSVs change during the build, include those changes in the PR.

## Testing Expectations

Use both Scala and PySpark coverage when behavior crosses Spark planning/runtime boundaries.

- Scala unit suites are best for config parsing, type conversion, decomposition, retry behavior, and file-format plumbing.
- PySpark integration tests are best for CPU/GPU result parity, fallback assertions, SQL syntax, end-to-end reads/writes, and Spark planner behavior.
- Prefer `assert_gpu_and_cpu_are_equal_collect` for parity and `assert_gpu_fallback_collect` for expected fallback.
- Add `@allow_non_gpu(...)` only for intentional CPU fragments.
- Test representative unsupported paths, not just the happy GPU path.
- Include literals and columns when the Spark function supports both.
- Include nulls, empty inputs, nested arrays/structs/maps, decimals, binary/string edge cases, and ANSI/non-ANSI behavior when applicable.
- For time-sensitive features, run non-UTC tests:
  ```bash
  TZ=Asia/Shanghai ./integration_tests/run_pyspark_from_build.sh
  TZ=America/Los_Angeles ./integration_tests/run_pyspark_from_build.sh
  ```

Targeted commands:

```bash
# Build the affected modules and generated docs/support data
mvn package -pl integration_tests,tests,tools -am -P 'individual,pre-merge' -Dbuildver=341 -Dmaven.scalastyle.skip=true -Drat.skip=true -DskipTests -Dmaven.scaladoc.skip

# Run one Scala suite
mvn package -pl tests -am -DwildcardSuites="com.nvidia.spark.rapids.ParquetWriterSuite"

# Run one or more PySpark integration tests
TESTS="string_test.py::test_replace" ./integration_tests/run_pyspark_from_build.sh
```

## PR Readiness

Before calling the feature done:

- Confirm registration says exactly what is supported and why unsupported cases fall back.
- Confirm docs and generated CSVs match the implementation.
- Confirm tests prove CPU/GPU parity and expected fallback.
- Check scalastyle and imports, especially after moving code into shims.
- Run `git diff --stat` and verify the change has the expected footprint for its feature type.
- Mention any intentional limitations in docs, fallback messages, and the PR description.

import json
import os
from typing import Any, Optional

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import DataProcessing, Inference, PromptProcessing
from eureka_ml_insights.core.eval_reporting import EvalReporting
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    ExtractUsageTransform,
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    SamplerTransform,
    SequenceTransform,
    MapStringsTransform,
    CopyColumn,
)
from eureka_ml_insights.data_utils.cruxeval_utils import (
    CRUXEvalGenerateQuestion,
)
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.metrics_base import ExactMatch
from eureka_ml_insights.metrics.reports import (
    BiLevelAggregator,
    BiLevelCountAggregator,
    CountAggregator,
)


class CRUXEval_PIPELINE(ExperimentConfig):
    """This class specifies the config for running CRUXEval benchmark on any model"""

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: str = None,
        n_repeats: int = 1,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:

        # --------------------------------------
        # * Data preprocessing
        # --------------------------------------
        # Prepare data for inference, apply transformation, or apply a Jinja prompt template.

        self.preprocessing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "cruxeval-org/cruxeval",
                    "split": "test",
                    "tasks": "default",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "output": "answer",
                                }
                            ),
                            AddColumn("question"),
                            CRUXEvalGenerateQuestion("code", "input", "question"),
                            SamplerTransform(sample_count=5, random_seed=99),
                            MultiplyTransform(n_repeats=int(n_repeats)),
                        ],
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/cruxeval_templates/zeroshot-v1.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # --------------------------------------
        # * Inference
        # --------------------------------------
        # Run model on any processed data

        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.preprocessing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=10,
        )

        # --------------------------------------
        # * Extract usage
        # --------------------------------------
        # Get token usage information

        self.usage_extraction_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ExtractUsageTransform(model_config),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_usage_extraction"),
        )

        # --------------------------------------
        # * Extract answer
        # --------------------------------------
        # Extract answer from raw model output

        self.postprocessing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.usage_extraction_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "answer": "ground_truth",
                                }
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # --------------------------------------
        # * Evaluation
        # --------------------------------------
        # Report metrics and aggregate results

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.postprocessing_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                # Aggregates across all repeats in one pool (no group_by)
                # - single overall pass@1 score for the entire dataset
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["ExactMatch_result"],
                        "normalize": True,
                        "filename_base": "ExactMatch",
                    },
                ),
                # Get average usage across all data points
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_Mean",
                        "agg_fn": "mean",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        pipeline_steps = [
            self.preprocessing_comp,
            self.inference_comp,
            self.usage_extraction_comp,
            self.postprocessing_comp,
            self.evalreporting_comp,
        ]

        if int(n_repeats) > 1:
            multirun_steps = self._configure_multirun_steps()
            pipeline_steps.extend(multirun_steps)

        return PipelineConfig(pipeline_steps, self.log_dir)


    def _configure_multirun_steps(self) -> list[Any]:
        """
        Builds and returns all additional aggregator configs and post-eval steps
        needed for multi-run analysis: separate/average of runs, best-of-n,
        worst-of-n, usage stats, and majority voting.
        """

        # Extend aggregator configs on the existing evalreporting_comp
        self.evalreporting_comp.aggregator_configs.extend(
            [
                # Separate run accuracy (pass@1 for each repeat)
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["ExactMatch_result"],
                        "group_by": "data_repeat_id",
                        "filename_base": "ExactMatch_SeparateRuns",
                        "normalize": True,
                    },
                ),
                # All-run accuracy (mean and std of pass@1 across repeats)
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": ["ExactMatch_result"],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "ExactMatch_AverageOfRuns",
                        "normalize": True,
                    },
                ),
                # Calculate usage stats per repeat
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_repeat_id",
                        "filename_base": "UsageCompletion_MeanofN",
                        "agg_fn": "mean",
                    },
                ),
                # Sums usage across all repeats for each data point
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["usage_completion"],
                        "first_groupby": "data_point_id",
                        "filename_base": "UsageCompletion_AllN",
                        "agg_fn": "sum",
                    },
                ),
            ]
        )

        # Convert "ExactMatch_result" from correct/incorrect => 1/0
        self.posteval_data_numeric_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.evalreporting_comp.output_dir, "metric_results.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            CopyColumn("ExactMatch_result", "ExactMatch_result_numeric"),
                            MapStringsTransform(
                                columns=["ExactMatch_result_numeric"],
                                mapping={"correct": "1", "incorrect": "0", "none": "NaN"},
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_posteval_numeric_output"),
        )

        # Best-of-n aggregator
        self.bestofn_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_numeric_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                # Measures fraction of data points solved by at least one attempt
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["ExactMatch_result_numeric"],
                        "first_groupby": "data_point_id",
                        "filename_base": "ExactMatch_BestOfN",
                        "agg_fn": "max",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_bestofn"),
        )

        # Worst-of-n aggregator
        self.worstofn_evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.posteval_data_numeric_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                # Measures fraction of data points correct on every attempt.
                AggregatorConfig(
                    BiLevelAggregator,
                    {
                        "column_names": ["ExactMatch_result_numeric"],
                        "first_groupby": "data_point_id",
                        "filename_base": "ExactMatch_WorstOfN",
                        "agg_fn": "min",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_worstofn"),
        )

        # Majority voting for multiple runs
        self.postprocessing_majorityvote_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "answer": "ground_truth",
                                }
                            ),
                            MajorityVoteTransform(id_col="data_point_id"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_postprocessing_output_majorityvote"),
        )

        self.evalreporting_majorityvote_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.postprocessing_majorityvote_comp.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": ["ExactMatch_result"],
                        "first_groupby": "data_point_id",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_majorityvote"),
        )

        return [
            self.posteval_data_numeric_comp,
            self.bestofn_evalreporting_comp,
            self.worstofn_evalreporting_comp,
            self.postprocessing_majorityvote_comp,
            self.evalreporting_majorityvote_comp,
        ]
    
'''
        # ====================================================
        # For multi-run evaluation
        # ====================================================
        # --------------------------------------
        # Aggregate the results (majority vote)
        # --------------------------------------

        self.postprocessing_comp = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    "answer": "ground_truth",
                                }
                            ),
                            MajorityVoteTransform(id_col="data_point_id"),
                            ColumnRename(
                                name_mapping={
                                    "model_output": "model_output_onerun",
                                    "majority_vote": "model_output",
                                }
                            ),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_addmv_output"),
        )

        # --------------------------------------
        # Compute accuracy
        # --------------------------------------

        self.postevalprocess_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing_addmv.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(
                    BiLevelCountAggregator,
                    {
                        "column_names": [
                            "ExactMatch_result",
                        ],
                        "first_groupby": "data_point_id",
                        "filename_base": "MajorityVote",
                        "normalize": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report_majorityVote"),
        )

'''


# =============================
# MUTATED CRUXEval BENCHMARK
# =============================


class CRUXEval_MUTATED_PIPELINE(CRUXEval_PIPELINE):
    """This class specifies the config for running mutated CRUXEval benchmark on any model"""

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: str = None,
        mutation_type: str = "Factual",
        n_repeats: Optional[int] = 1,
        sample_count: Optional[int] = None,
        prompt_template_name: Optional[str] = None,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:

        pipeline = super().configure_pipeline(
            model_config=model_config,
            resume_from=resume_from,
            n_repeats=n_repeats,
        )

        # -----------------------------
        # Retrieve data and pre-process
        # -----------------------------
        with open("data/paths_cruxeval.json", "r") as f:
            DATA_PATHS = json.load(f)
        path = DATA_PATHS[mutation_type]

        self.preprocessing_comp.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": path,
                "split": "test",
                "load_data_from_disk": True,
                "transform": SequenceTransform([]),
            },
        )

        if sample_count is not None:
            self.preprocessing_comp.data_reader_config.init_args["transform"].transforms.append(
                SamplerTransform(sample_count=int(sample_count), random_seed=99)
            )

        self.preprocessing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=int(n_repeats)),
        )

        if prompt_template_name is not None:
            self.preprocessing_comp.prompt_template_path = os.path.join(
                os.path.dirname(__file__), f"../prompt_templates/cruxeval_templates/{prompt_template_name}.jinja"
            )

        return pipeline


# def __main__():
#     data_reader = HFDataReader(
#         path = "cruxeval-org/cruxeval",
#         split =  "test",
#         tasks = "default",
#     )

#     df = data_reader._load_dataset()
#     print(df.head())

#     mutation_type = 'Factual'
#     with open("data/paths_cruxeval.json", "r") as f:
#         DATA_PATHS = json.load(f)
#     path = DATA_PATHS[mutation_type]
#     data_reader_mutated = HFDataReader(
#         path = path,
#         split =  "test",
#         tasks = None,
#         load_data_from_disk = True,
#     )
#     df_mutated = data_reader_mutated._load_dataset()
#     print(df_mutated.head())

#     breakpoint()


# if __name__ == "__main__":
#     __main__()

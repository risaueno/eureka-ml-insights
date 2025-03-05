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
    HFDataReader,
    MajorityVoteTransform,
    MultiplyTransform,
    SamplerTransform,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.cruxeval_utils import (
    CRUXEvalGenerateQuestion,
)
from eureka_ml_insights.data_utils.data import DataLoader
from eureka_ml_insights.metrics.metrics_base import ExactMatch
from eureka_ml_insights.metrics.reports import (
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
        # Data preprocessing
        # --------------------------------------
        # * PromptProcessing:
        # prepare data for inference, apply transformation, or apply a Jinja prompt template.

        self.data_processing_comp = PromptProcessingConfig(
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
                            # SamplerTransform(sample_count=5, random_seed=99),
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
        # Inference
        # --------------------------------------
        # * Inference:  run your model on any processed data,
        # for example running inference on the model subject to evaluation,
        # or another model that is involved in the evaluation pipeline as an evaluator or judge.
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                DataLoader,
                {"path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl")},
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=10,
        )

        # --------------------------------------
        # Extract answer
        # --------------------------------------
        # * DataProcessing: you can use this component to to post-process the model outputs.
        self.data_post_processing = DataProcessingConfig(
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
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        # --------------------------------------
        # Evaluation (compute accuracy)
        # --------------------------------------
        # * EvalReporting: evaluate the model outputs using various metrics, aggregators
        # and visualizers, and generate a report.
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.data_post_processing.output_dir, "transformed_data.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ExactMatch),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": [
                            "ExactMatch_result",
                        ],
                        "normalize": True,
                        "filename_base": "ExactMatch",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        # ====================================================
        # For multi-run evaluation
        # ====================================================
        # --------------------------------------
        # Aggregate the results (majority vote)
        # --------------------------------------

        self.data_post_processing_addmv = DataProcessingConfig(
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

        # Output pipeline config in order
        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
                self.data_post_processing_addmv,
                self.postevalprocess_comp,
            ],
            self.log_dir,
        )


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

        self.data_processing_comp.data_reader_config = DataSetConfig(
            HFDataReader,
            {
                "path": path,
                "split": "test",
                "load_data_from_disk": True,
                "transform": SequenceTransform([]),
            },
        )

        if sample_count is not None:
            self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
                SamplerTransform(sample_count=int(sample_count), random_seed=99)
            )

        self.data_processing_comp.data_reader_config.init_args["transform"].transforms.append(
            MultiplyTransform(n_repeats=int(n_repeats)),
        )

        if prompt_template_name is not None:
            self.data_processing_comp.prompt_template_path = os.path.join(
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

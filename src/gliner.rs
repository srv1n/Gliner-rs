use crate::decoder::BaseDecoder;
use crate::decoder::SpanDecoder;
use crate::model::Model;
use crate::onnxwrapper::ONNXWrapper;
use crate::processor::SpanProcessor;
use crate::processor::{Processor, WhitespaceTokenSplitter};
use crate::types::InferenceResultSingle;
use crate::types::{EntityResult, InferenceResultMultiple, RawInferenceResult};
use anyhow::Result;
use std::collections::HashMap;
use tokenizers::Tokenizer;
#[derive(Clone)]
pub struct InitConfig {
    pub tokenizer_path: String,
    pub model_path: String,
    pub num_threads: Option<usize>,

    // pub onnx_settings: ONNXSettings,
    pub max_width: Option<usize>,
}

pub struct Gliner {
    config: InitConfig,
    model: Option<Model>,
}

impl Gliner {
    pub fn new(config: InitConfig) -> Self {
        Gliner {
            config,
            model: None,
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        let tokenizer = Tokenizer::from_file(self.config.tokenizer_path.clone())
            .map_err(|e| anyhow::anyhow!("Error initializing tokenizer: {}", e))?;
        let onnx_wrapper =
            ONNXWrapper::new(self.config.model_path.clone(), self.config.num_threads);

        let processor = SpanProcessor::new(
            HashMap::from([(
                "max_width".to_string(),
                self.config.max_width.unwrap_or(12).to_string(),
            )]),
            tokenizer,
        );
        let decoder = SpanDecoder::new(HashMap::from([(
            "max_width".to_string(),
            self.config.max_width.unwrap_or(12).to_string(),
        )]));

        let model = Model::new(
            HashMap::from([(
                "max_width".to_string(),
                self.config.max_width.unwrap_or(12).to_string(),
            )]),
            processor,
            Box::new(decoder),
            onnx_wrapper.session,
        );

        self.model = Some(model);
        self.model.as_mut().unwrap().initialize().await
    }

    pub async fn inference(
        &self,
        texts: &[String],
        entities: &[&str],
        flat_ner: bool,
        threshold: f32,
    ) -> Result<InferenceResultMultiple> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model is not initialized. Call initialize() first."))?;
        let result: RawInferenceResult = model
            .inference(texts, entities, flat_ner, threshold)
            .await?;
        Ok(self.map_raw_result_to_response(result))
    }

    // pub async fn inference_with_chunking(
    //     &self,
    //     texts: &[String],
    //     entities: &[String],
    //     flat_ner: bool,
    //     threshold: f32,
    // ) -> Result<InferenceResultMultiple> {
    //     let model = self
    //         .model
    //         .as_ref()
    //         .ok_or_else(|| anyhow::anyhow!("Model is not initialized. Call initialize() first."))?;
    //     let result = model
    //         .inference_with_chunking(texts, entities, flat_ner, threshold)
    //         .await?;
    //     Ok(self.map_raw_result_to_response(result))
    // }

    // fn map_raw_result_to_response(
    //     &self,
    //     raw_result: RawInferenceResult,
    // ) -> InferenceResultMultiple {
    //     raw_result
    //         .into_iter()
    //         .map(|individual_result| {
    //             individual_result
    //                 .into_iter()
    //                 .map(|(span_text, start, end, label, score)| EntityResult {
    //                     span_text,
    //                     start,
    //                     end,
    //                     label,
    //                     score,
    //                 })
    //                 .collect()
    //         })
    //         .collect()
    // }

    fn map_raw_result_to_response(
        &self,
        raw_result: RawInferenceResult,
    ) -> Vec<InferenceResultSingle> {
        raw_result
            .into_iter()
            .map(|individual_result| InferenceResultSingle {
                entities: individual_result
                    .into_iter()
                    .map(|(span_text, start, end, label, score)| EntityResult {
                        span_text,
                        start,
                        end,
                        label,
                        score,
                    })
                    .collect(),
            })
            .collect()
    }
}

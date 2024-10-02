use crate::processor::SpanProcessor;
use crate::types::RawInferenceResult;
use crate::{decoder::Decoder, processor::Processor};
use anyhow::{Ok, Result};
use ndarray::Array;
use ort::SessionInputValue;
use ort::{Session, Value};
use std::borrow::Cow;
use std::collections::HashMap;
pub struct Model {
    config: HashMap<String, String>,
    processor: SpanProcessor,
    decoder: Box<dyn Decoder>,
    session: Session,
}

impl Model {
    pub fn new(
        config: HashMap<String, String>,
        processor: SpanProcessor,
        decoder: Box<dyn Decoder>,
        session: Session,
    ) -> Self {
        Model {
            config,
            processor,
            decoder,
            session,
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        // If there's any initialization needed, do it here
        Ok(())
    }

    // fn prepare_inputs(
    //     &self,
    //     batch: &HashMap<String, Vec<Vec<usize>>>,
    // ) -> Result<Vec<(Cow<'_, str>, SessionInputValue<'_>)>> {
    //     let batch_size = batch["inputsIds"].len();
    //     let num_tokens = batch["inputsIds"][0].len();
    //     let num_spans = batch["spanIdxs"][0].len();

    //     let create_tensor = |data: &Vec<Vec<usize>>, shape: &[usize]| -> Result<Value> {
    //         let flat_data: Vec<i64> = data.iter().flatten().map(|&x| x as i64).collect();
    //         Ok(Value::from_array(
    //             Array::from_shape_vec(shape.to_vec(), flat_data)?.into_dyn(),
    //         ))
    //     };

    //     let mut session_input = vec![
    //         (
    //             "input_ids".into(),
    //             create_tensor(&batch["inputsIds"], &[batch_size, num_tokens])?.into(),
    //         ),
    //         (
    //             "attention_mask".into(),
    //             create_tensor(&batch["attentionMasks"], &[batch_size, num_tokens])?.into(),
    //         ),
    //         (
    //             "words_mask".into(),
    //             create_tensor(&batch["wordsMasks"], &[batch_size, num_tokens])?.into(),
    //         ),
    //         (
    //             "text_lengths".into(),
    //             create_tensor(&batch["textLengths"], &[batch_size, 1])?.into(),
    //         ),
    //         (
    //             "span_idx".into(),
    //             create_tensor(&batch["spanIdxs"], &[batch_size, num_spans, 2])?.into(),
    //         ),
    //     ];

    //     let span_masks: Vec<i64> = batch["spanMasks"]
    //         .iter()
    //         .flatten()
    //         .map(|&x| x as i64)
    //         .collect();
    //     session_input.push((
    //         "span_mask".into(),
    //         Value::from_array(Array::from_shape_vec([batch_size, num_spans], span_masks)?)?.into(),
    //     ));

    //     Ok(session_input)
    // }

    pub async fn inference(
        &self,
        texts: &[String],
        entities: &[&str],
        flat_ner: bool,
        threshold: f32,
    ) -> Result<RawInferenceResult> {
        // ) -> Result<()> {
        let (
            session_input,
            id_to_class,
            batch_tokens,
            batch_words_start_idx,
            batch_words_end_idx,
            text_lengths,
        ) = self.processor.prepare_batch(texts, entities);
        // let (session_input, _, _) = self.processor.prepare_inputs(&batch);

        // println!("Session input {:#?}", self.session.inputs);
        // Run the model
        let outputs = self.session.run(session_input).unwrap();
        // println!("Outputs {:?}", outputs.);
        let logits = outputs[0].try_extract_tensor::<f32>()?;
        // println!("Logits {:?}", logits);
        // Ok(())
        // Extract logits
        // let logits = outputs[0].try_extract::<f32>()?;
        // let logits_shape = logits.shape();
        // println!("Logits shape {:?}", logits_shape);

        let batch_size = batch_tokens.len();
        // println!("Batch size {:?}", batch_size);
        let input_length = *text_lengths.iter().max().unwrap();
        // println!("Input length {:?}", input_length);
        let max_width = self.config["max_width"].parse::<usize>().unwrap();
        let num_entities = entities.len();
        // println!("Num entities {:?}", num_entities);
        let batch_ids: Vec<usize> = (0..batch_size).collect();

        let decoded_spans: RawInferenceResult = self.decoder.decode(
            batch_size,
            input_length,
            max_width,
            num_entities,
            &texts,
            &batch_ids,
            &batch_words_start_idx,
            &batch_words_end_idx,
            &id_to_class,
            logits.as_slice().unwrap(),
            flat_ner,
            threshold,
            false, // multi_label is not used in the TypeScript version
        );

        // Ok(())

        Ok(decoded_spans)
    }

    // pub async fn inference_with_chunking(
    //     &self,
    //     texts: &[String],
    //     entities: &[String],
    //     flat_ner: bool,
    //     threshold: f32,
    // ) -> Result<RawInferenceResult> {
    //     let (class_to_id, id_to_class) = self.processor.create_mappings(entities);

    //     let mut batch_ids = Vec::new();
    //     let mut batch_tokens = Vec::new();
    //     let mut batch_words_start_idx = Vec::new();
    //     let mut batch_words_end_idx = Vec::new();

    //     for (id, text) in texts.iter().enumerate() {
    //         let (tokens, words_start_idx, words_end_idx) = self.processor.tokenize_text(text);
    //         let num_sub_batches = (tokens.len() as f32 / 512.0).ceil() as usize;

    //         for i in 0..num_sub_batches {
    //             let start = i * 512;
    //             let end = (start + 512).min(tokens.len());

    //             batch_ids.push(id);
    //             batch_tokens.push(tokens[start..end].to_vec());
    //             batch_words_start_idx.push(words_start_idx[start..end].to_vec());
    //             batch_words_end_idx.push(words_end_idx[start..end].to_vec());
    //         }
    //     }

    //     let num_batches = (batch_ids.len() as f32 / 16.0).ceil() as usize;

    //     let mut final_decoded_spans: RawInferenceResult = vec![vec![]; texts.len()];

    //     for batch_id in 0..num_batches {
    //         let start = batch_id * 16;
    //         let end = (start + 16).min(batch_ids.len());

    //         let curr_batch_tokens = &batch_tokens[start..end];
    //         let curr_batch_words_start_idx = &batch_words_start_idx[start..end];
    //         let curr_batch_words_end_idx = &batch_words_end_idx[start..end];
    //         let curr_batch_ids = &batch_ids[start..end];

    //         let (input_tokens, text_lengths, prompt_lengths) = self
    //             .processor
    //             .prepare_text_inputs(curr_batch_tokens, entities);

    //         let (mut inputs_ids, mut attention_masks, mut words_masks) = self
    //             .processor
    //             .encode_inputs(&input_tokens, Some(&prompt_lengths));

    //         self.processor.pad_array(&mut inputs_ids, 2);
    //         self.processor.pad_array(&mut attention_masks, 2);
    //         self.processor.pad_array(&mut words_masks, 2);

    //         let (mut span_idxs, mut span_masks) = self
    //             .processor
    //             .prepare_spans(curr_batch_tokens, self.config["max_width"].parse().unwrap());

    //         self.processor.pad_array(&mut span_idxs, 3);
    //         self.processor.pad_array(&mut span_masks, 2);

    //         let mut batch = HashMap::new();
    //         batch.insert("inputsIds".to_string(), inputs_ids);
    //         batch.insert("attentionMasks".to_string(), attention_masks);
    //         batch.insert("wordsMasks".to_string(), words_masks);
    //         batch.insert("textLengths".to_string(), text_lengths);
    //         batch.insert("spanIdxs".to_string(), span_idxs);
    //         batch.insert("spanMasks".to_string(), span_masks);
    //         batch.insert("idToClass".to_string(), id_to_class.clone());
    //         batch.insert("batchTokens".to_string(), curr_batch_tokens.to_vec());
    //         batch.insert(
    //             "batchWordsStartIdx".to_string(),
    //             curr_batch_words_start_idx.to_vec(),
    //         );
    //         batch.insert(
    //             "batchWordsEndIdx".to_string(),
    //             curr_batch_words_end_idx.to_vec(),
    //         );

    //         let feeds = self.prepare_inputs(&batch)?;
    //         let outputs = self.session.run(feeds)?;
    //         let logits = outputs[0].try_extract::<f32>()?;
    //         let logits_shape = logits.shape();

    //         let batch_size = batch["batchTokens"].len();
    //         let input_length = *batch["textLengths"].iter().flatten().max().unwrap();
    //         let max_width = self.config["max_width"].parse::<usize>().unwrap();
    //         let num_entities = entities.len();

    //         let decoded_spans = self.decoder.decode(
    //             batch_size,
    //             input_length,
    //             max_width,
    //             num_entities,
    //             &batch["batchTokens"],
    //             curr_batch_ids,
    //             &batch["batchWordsStartIdx"],
    //             &batch["batchWordsEndIdx"],
    //             &batch["idToClass"],
    //             logits.as_slice().unwrap(),
    //             flat_ner,
    //             threshold,
    //             false, // multi_label is not used in the TypeScript version
    //         );

    //         for (i, &original_text_id) in curr_batch_ids.iter().enumerate() {
    //             final_decoded_spans[original_text_id].extend_from_slice(&decoded_spans[i]);
    //         }
    //     }

    //     Ok(final_decoded_spans)
    // }
}

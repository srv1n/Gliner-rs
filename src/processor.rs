use ndarray::{Array1, Array2, Array3};
use ort::{SessionInputValue, Value};
use regex::Regex;
use std::{borrow::Cow, collections::HashMap};
use tokenizers::Tokenizer;

use ndarray::{Array, ArrayD, Axis, IxDyn};

pub struct WhitespaceTokenSplitter {
    whitespace_pattern: Regex,
}

impl WhitespaceTokenSplitter {
    pub fn new() -> Self {
        WhitespaceTokenSplitter {
            whitespace_pattern: Regex::new(r"\w+(?:[-_]\w+)*|\S").unwrap(),
        }
    }

    pub fn call<'a>(&'a self, text: &'a str) -> impl Iterator<Item = (String, usize, usize)> + 'a {
        self.whitespace_pattern
            .find_iter(text)
            .map(|m| (m.as_str().to_string(), m.start(), m.end()))
    }
}

pub struct Processor {
    config: HashMap<String, String>,
    tokenizer: Tokenizer,
    words_splitter: WhitespaceTokenSplitter,
}

impl Processor {
    pub fn new(config: HashMap<String, String>, tokenizer: Tokenizer) -> Self {
        Processor {
            config,
            tokenizer,
            words_splitter: WhitespaceTokenSplitter::new(),
        }
    }

    pub fn tokenize_text(&self, text: &str) -> (Vec<String>, Vec<usize>, Vec<usize>) {
        let mut tokens = Vec::new();
        let mut words_start_idx = Vec::new();
        let mut words_end_idx = Vec::new();

        for (token, start, end) in self.words_splitter.call(text) {
            tokens.push(token);
            words_start_idx.push(start);
            words_end_idx.push(end);
        }

        (tokens, words_start_idx, words_end_idx)
    }

    pub fn batch_tokenize_text(
        &self,
        texts: &[String],
    ) -> (Vec<Vec<String>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut batch_tokens = Vec::new();
        let mut batch_words_start_idx = Vec::new();
        let mut batch_words_end_idx = Vec::new();

        for text in texts {
            let (tokens, words_start_idx, words_end_idx) = self.tokenize_text(text);
            batch_tokens.push(tokens);
            batch_words_start_idx.push(words_start_idx);
            batch_words_end_idx.push(words_end_idx);
        }
        // println!("batch_tokens {:?}", batch_tokens);
        (batch_tokens, batch_words_start_idx, batch_words_end_idx)
    }

    pub fn create_mappings(
        &self,
        classes: &[&str],
    ) -> (HashMap<String, usize>, HashMap<usize, String>) {
        let mut class_to_id = HashMap::new();
        let mut id_to_class = HashMap::new();

        for (index, class_name) in classes.iter().enumerate() {
            let id = index;
            class_to_id.insert(class_name.to_string(), id);
            id_to_class.insert(id, class_name.to_string());
        }

        (class_to_id, id_to_class)
    }

    pub fn prepare_text_inputs(
        &self,
        tokens: &[Vec<String>],
        entities: &[&str],
    ) -> (Vec<Vec<String>>, Vec<usize>, Vec<usize>) {
        let mut input_texts = Vec::new();
        let mut prompt_lengths = Vec::new();
        let mut text_lengths = Vec::new();

        for text in tokens {
            text_lengths.push(text.len());

            let mut input_text = Vec::new();
            for ent in entities {
                input_text.push("<<ENT>>".to_string());
                input_text.push(ent.to_string());
            }
            input_text.push("<<SEP>>".to_string());
            let prompt_length = input_text.len();
            prompt_lengths.push(prompt_length);
            input_text.extend_from_slice(text);
            input_texts.push(input_text);
        }

        (input_texts, text_lengths, prompt_lengths)
    }

    pub fn encode_inputs(
        &self,
        texts: &[Vec<String>],
        prompt_lengths: Option<&[usize]>,
    ) -> (Vec<Vec<u32>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut words_masks = Vec::new();
        let mut inputs_ids = Vec::new();
        let mut attention_masks = Vec::new();

        for (id, text) in texts.iter().enumerate() {
            let prompt_length = prompt_lengths.map_or(0, |pl| pl[id]);
            let mut words_mask = vec![0];
            let mut input_ids = vec![1];
            let mut attention_mask = vec![1];

            let mut c = 1;
            for (word_id, word) in text.iter().enumerate() {
                let word_tokens = self.tokenizer.encode(word.as_str(), false).unwrap();
                let word_tokens = word_tokens.get_ids();
                for (token_id, &token) in word_tokens.iter().enumerate() {
                    attention_mask.push(1);
                    if word_id < prompt_length {
                        words_mask.push(0);
                    } else if token_id == 0 {
                        words_mask.push(c);
                        c += 1;
                    } else {
                        words_mask.push(0);
                    }
                    input_ids.push(token);
                }
            }
            words_mask.push(0);
            input_ids.push(128003);
            attention_mask.push(1);

            words_masks.push(words_mask);
            inputs_ids.push(input_ids);
            attention_masks.push(attention_mask);
        }

        (inputs_ids, attention_masks, words_masks)
    }

    // pub fn pad_array<T: Clone + Default>(&self, arr: &mut Vec<Vec<T>>, dimensions: usize) {
    //     if dimensions < 2 || dimensions > 3 {
    //         panic!("Only 2D and 3D arrays are supported");
    //     }

    //     let max_length = arr.iter().map(|sub_arr| sub_arr.len()).max().unwrap_or(0);

    //     for sub_arr in arr.iter_mut() {
    //         let pad_count = max_length - sub_arr.len();
    //         sub_arr.extend(std::iter::repeat(T::default()).take(pad_count));
    //     }
    // }
    // pub fn pad_array(
    //     &self,
    //     arr: &Vec<Vec<Vec<usize>>>,
    //     dimensions: usize,
    // ) -> Result<Vec<Vec<Vec<usize>>>, String> {
    //     if dimensions < 2 || dimensions > 3 {
    //         return Err("Only 2D and 3D arrays are supported".to_string());
    //     }

    //     let max_length = arr.iter().map(|sub_arr| sub_arr.len()).max().unwrap_or(0);
    //     let final_dim = if dimensions == 3 { arr[0][0].len() } else { 0 };

    //     let result = arr
    //         .iter()
    //         .map(|sub_arr| {
    //             let pad_count = max_length - sub_arr.len();
    //             let padding = vec![
    //                 if dimensions == 3 {
    //                     vec![0; final_dim]
    //                 } else {
    //                     vec![]
    //                 };
    //                 pad_count
    //             ];
    //             let mut new_sub_arr = sub_arr.clone();
    //             new_sub_arr.extend(padding);
    //             new_sub_arr
    //         })
    //         .collect();

    //     Ok(result)
    // }
}

pub struct SpanProcessor {
    processor: Processor,
}

impl SpanProcessor {
    pub fn new(config: HashMap<String, String>, tokenizer: Tokenizer) -> Self {
        SpanProcessor {
            processor: Processor::new(config, tokenizer),
        }
    }

    pub fn prepare_spans(
        &self,
        batch_tokens: &[Vec<String>],
        max_width: usize,
    ) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<bool>>) {
        let mut span_idxs = Vec::new();
        let mut span_masks = Vec::new();

        for tokens in batch_tokens {
            let text_length = tokens.len();
            let mut span_idx = Vec::new();
            let mut span_mask = Vec::new();

            for i in 0..text_length {
                for j in 0..max_width {
                    let end_idx = (i + j).min(text_length - 1);
                    span_idx.push(vec![i, end_idx]);
                    span_mask.push(end_idx < text_length);
                }
            }

            span_idxs.push(span_idx);
            span_masks.push(span_mask);
        }

        (span_idxs, span_masks)
    }

    pub fn prepare_batch(
        &self,
        texts: &[String],
        entities: &[&str],
    ) -> (
        Vec<(Cow<'_, str>, SessionInputValue<'_>)>,
        HashMap<usize, String>,
        Vec<Vec<String>>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        Vec<usize>,
    ) {
        let (batch_tokens, batch_words_start_idx, batch_words_end_idx) =
            self.processor.batch_tokenize_text(texts);
        let (_, id_to_class) = self.processor.create_mappings(entities);
        let (input_tokens, text_lengths, prompt_lengths) =
            self.processor.prepare_text_inputs(&batch_tokens, entities);

        let (mut inputs_ids, mut attention_masks, mut words_masks) = self
            .processor
            .encode_inputs(&input_tokens, Some(&prompt_lengths));

        inputs_ids = pad_array(&inputs_ids);
        let batch_size = inputs_ids.len();
        attention_masks = pad_array(&attention_masks);
        words_masks = pad_array(&words_masks);

        // self.processor.pad_array(&mut inputs_ids, 2);
        // self.processor.pad_array(&mut attention_masks, 2);
        // self.processor.pad_array(&mut words_masks, 2);

        let max_width = self.processor.config["max_width"].parse().unwrap_or(12);
        let (mut span_idxs, mut span_masks) = self.prepare_spans(&batch_tokens, max_width);

        // self.processor.pad_array(&mut span_idxs, 3);
        // self.processor.pad_array(&mut span_masks, 2);
        span_idxs = pad_array_3d(&span_idxs);
        span_masks = pad_array(&span_masks);

        let input_ids_insert: Array2<i64> = Array2::from_shape_vec(
            (inputs_ids.len(), inputs_ids[0].len()),
            inputs_ids
                .iter()
                .flat_map(|row| row.iter().map(|&id| id as i64))
                .collect(),
        )
        .unwrap();
        let attention_masks_insert: Array2<i64> = Array2::from_shape_vec(
            (attention_masks.len(), attention_masks[0].len()),
            attention_masks
                .iter()
                .flat_map(|row| row.iter().map(|&id| id as i64))
                .collect(),
        )
        .unwrap();
        let words_masks_insert: Array2<i64> = Array2::from_shape_vec(
            (words_masks.len(), words_masks[0].len()),
            words_masks
                .iter()
                .flat_map(|row| row.iter().map(|&id| id as i64))
                .collect(),
        )
        .unwrap();
        // let text_lengths_insert: Array1<i64> =
        //     Array1::from_vec(text_lengths.iter().map(|&len| len as i64).collect());
        let text_lengths_insert: Array2<i64> = Array2::from_shape_vec(
            (batch_size, 1),
            text_lengths.iter().map(|&len| len as i64).collect(),
        )
        .expect("Failed to reshape text_lengths");

        // println!("Span_idxs {:?}", span_idxs);
        // println!(
        //     "Span_idxs dimensions: {} x {} x {}",
        //     span_idxs.len(),
        //     span_idxs[0].len(),
        //     span_idxs[0][0].len()
        // );
        // let total_elements: usize = span_idxs
        //     .iter()
        //     .flat_map(|row| row.iter().map(|sub_row| sub_row.len()))
        //     .sum();
        // println!("Total elements in span_idxs: {}", total_elements);

        let span_idxs_insert: Array3<i64> = Array3::from_shape_vec(
            (span_idxs.len(), span_idxs[0].len(), span_idxs[0][0].len()),
            span_idxs
                .iter()
                .flat_map(|row| {
                    row.iter()
                        .flat_map(|sub_row| sub_row.iter().map(|&id| id as i64))
                })
                .collect(),
        )
        .unwrap();
        let span_masks_insert: Array2<bool> = Array2::from_shape_vec(
            (span_masks.len(), span_masks[0].len()),
            span_masks
                .iter()
                .flat_map(|row| row.iter().map(|&id| id))
                .collect(),
        )
        .unwrap();
        // let id_to_class_insert: Array2<i64> = Array2::from_shape_vec(
        //     (id_to_class.len(), 2),
        //     id_to_class
        //         .iter()
        //         .flat_map(|(_, &id)| vec![id as i64, 0])
        //         .collect(),
        // )
        // .unwrap();
        // let batch_tokens_insert: Array2<i64> = Array2::from_shape_vec(
        //     (batch_tokens.len(), batch_tokens[0].len()),
        //     batch_tokens
        //         .iter()
        //         .flat_map(|row| row.iter().map(|&id| id as i64))
        //         .collect(),
        // )
        // .unwrap();
        // batch_words_start_idx = pad_array(&batch_words_start_idx);

        // let batch_words_start_idx_insert: Array2<i64> = Array2::from_shape_vec(
        //     (batch_words_start_idx.len(), batch_words_start_idx[0].len()),
        //     batch_words_start_idx
        //         .iter()
        //         .flat_map(|row| row.iter().map(|&id| id as i64))
        //         .collect(),
        // )
        // .unwrap();
        // let batch_words_end_idx_insert: Array2<i64> = Array2::from_shape_vec(
        //     (batch_words_end_idx.len(), batch_words_end_idx[0].len()),
        //     batch_words_end_idx
        //         .iter()
        //         .flat_map(|row| row.iter().map(|&id| id as i64))
        //         .collect(),
        // )
        // .unwrap();

        let result = ort::inputs![
            "input_ids" => Value::from_array(input_ids_insert.clone())?,
            "attention_mask" => Value::from_array(attention_masks_insert.clone())?,
            "words_mask" => Value::from_array(words_masks_insert.clone())?,
            "text_lengths" => Value::from_array(text_lengths_insert.clone())?,
            "span_idx" => Value::from_array(span_idxs_insert.clone())?,
            "span_mask" => Value::from_array(span_masks_insert.clone())?,
            // "id_to_class" => Value::from_array(id_to_class_insert.clone())?,
            // "batch_tokens" => Value::from_array(batch_tokens_insert.clone())?,
            // "batch_words_start_idx" => Value::from_array(batch_words_start_idx_insert.clone())?,
            // "batch_words_end_idx" => Value::from_array(batch_words_end_idx_insert.clone())?,
        ]
        .unwrap();
        // result.insert("inputs_ids".to_string(), inputs_ids);
        // result.insert("attention_masks".to_string(), attention_masks);
        // result.insert("words_masks".to_string(), words_masks);
        // result.insert("text_lengths".to_string(), vec![text_lengths]);
        // result.insert(
        //     "span_idxs".to_string(),
        //     span_idxs.into_iter().flatten().collect(),
        // );
        // let span_masks_usize: Vec<Vec<usize>> = span_masks
        //     .into_iter()
        //     .map(|v| v.into_iter().map(|b| b as usize).collect())
        //     .collect();
        // result.insert("span_masks".to_string(), span_masks_usize);
        // Note: id_to_class, batch_tokens, batch_words_start_idx, and batch_words_end_idx are not included
        // as they are not easily representable in the same structure. You may need to handle these separately.

        (
            result,
            id_to_class,
            batch_tokens,
            batch_words_start_idx,
            batch_words_end_idx,
            text_lengths,
        )
    }
}

// You'll need to implement this trait for your specific tokenizer

// ... existing code ...

pub trait Padable: Clone + Default {}
impl Padable for usize {}
impl Padable for bool {}
impl Padable for u32 {}

pub fn pad_array<T: Padable>(arr: &[Vec<T>]) -> Vec<Vec<T>> {
    let max_length = arr.iter().map(|sub_arr| sub_arr.len()).max().unwrap_or(0);

    arr.iter()
        .map(|sub_arr| {
            let mut new_sub_arr = sub_arr.clone();
            new_sub_arr.resize(max_length, T::default());
            new_sub_arr
        })
        .collect()
}

pub fn pad_array_3d<T: Padable>(arr: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
    let max_length = arr.iter().map(|sub_arr| sub_arr.len()).max().unwrap_or(0);
    let inner_length = arr
        .get(0)
        .and_then(|first| first.get(0).map(|inner| inner.len()))
        .unwrap_or(0);

    arr.iter()
        .map(|sub_arr| {
            let mut new_sub_arr = sub_arr.clone();
            while new_sub_arr.len() < max_length {
                new_sub_arr.push(vec![T::default(); inner_length]);
            }
            new_sub_arr
        })
        .collect()
}

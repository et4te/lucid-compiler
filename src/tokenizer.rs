//! Tokenizer for Text Processing
//!
//! Provides character-level and BPE tokenization for language models.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

/// Special tokens
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;
pub const BOS_ID: usize = 2;
pub const EOS_ID: usize = 3;

/// Tokenizer trait for different tokenization strategies
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

/// Character-level tokenizer - simplest approach
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
    vocab_size: usize,
}

impl CharTokenizer {
    /// Create a new character tokenizer with ASCII + common characters
    pub fn new() -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Reserve special tokens
        let special_chars = ['\0', '\x01', '\x02', '\x03']; // PAD, UNK, BOS, EOS
        for (i, &c) in special_chars.iter().enumerate() {
            char_to_id.insert(c, i);
            id_to_char.insert(i, c);
        }

        // Add printable ASCII characters
        let mut next_id = 4;
        for c in ' '..='~' {
            char_to_id.insert(c, next_id);
            id_to_char.insert(next_id, c);
            next_id += 1;
        }

        // Add newline and tab
        char_to_id.insert('\n', next_id);
        id_to_char.insert(next_id, '\n');
        next_id += 1;

        char_to_id.insert('\t', next_id);
        id_to_char.insert(next_id, '\t');
        next_id += 1;

        CharTokenizer {
            char_to_id,
            id_to_char,
            vocab_size: next_id,
        }
    }

    /// Create from a text corpus (learns vocabulary)
    pub fn from_corpus(text: &str) -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Reserve special tokens
        char_to_id.insert('\0', PAD_ID);
        id_to_char.insert(PAD_ID, '\0');
        char_to_id.insert('\x01', UNK_ID);
        id_to_char.insert(UNK_ID, '\x01');
        char_to_id.insert('\x02', BOS_ID);
        id_to_char.insert(BOS_ID, '\x02');
        char_to_id.insert('\x03', EOS_ID);
        id_to_char.insert(EOS_ID, '\x03');

        let mut next_id = 4;
        for c in text.chars() {
            if !char_to_id.contains_key(&c) {
                char_to_id.insert(c, next_id);
                id_to_char.insert(next_id, c);
                next_id += 1;
            }
        }

        CharTokenizer {
            char_to_id,
            id_to_char,
            vocab_size: next_id,
        }
    }

    /// Add BOS token to sequence
    pub fn add_bos(&self, tokens: &mut Vec<usize>) {
        tokens.insert(0, BOS_ID);
    }

    /// Add EOS token to sequence
    pub fn add_eos(&self, tokens: &mut Vec<usize>) {
        tokens.push(EOS_ID);
    }
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&UNK_ID))
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&id| {
                if id == PAD_ID || id == BOS_ID || id == EOS_ID {
                    None
                } else {
                    self.id_to_char.get(&id).copied()
                }
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Simple BPE (Byte Pair Encoding) tokenizer
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    id_to_token: HashMap<usize, String>,
    /// Merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Vocabulary size
    vocab_size: usize,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer from a corpus
    pub fn train(corpus: &str, vocab_size: usize, min_frequency: usize) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut merges = Vec::new();

        // Add special tokens
        let special = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN];
        for (i, &tok) in special.iter().enumerate() {
            token_to_id.insert(tok.to_string(), i);
            id_to_token.insert(i, tok.to_string());
        }

        // Initialize with character-level tokens
        let mut next_id = 4;
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

        // Split corpus into words and count
        for word in corpus.split_whitespace() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            // Add each character to vocab
            for c in &chars {
                if !token_to_id.contains_key(c) {
                    token_to_id.insert(c.clone(), next_id);
                    id_to_token.insert(next_id, c.clone());
                    next_id += 1;
                }
            }
            // Add end-of-word marker
            let mut word_tokens = chars;
            word_tokens.push("</w>".to_string());
            *word_freqs.entry(word_tokens).or_insert(0) += 1;
        }

        // Add end-of-word marker to vocab
        if !token_to_id.contains_key("</w>") {
            token_to_id.insert("</w>".to_string(), next_id);
            id_to_token.insert(next_id, "</w>".to_string());
            next_id += 1;
        }

        // BPE merging loop
        while next_id < vocab_size {
            // Count pairs
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
            for (word, freq) in &word_freqs {
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .filter(|(_, &freq)| freq >= min_frequency)
                .max_by_key(|(_, &freq)| freq);

            let (best, _) = match best_pair {
                Some((pair, freq)) => (pair.clone(), *freq),
                None => break,
            };

            // Create merged token
            let merged = format!("{}{}", best.0, best.1);
            token_to_id.insert(merged.clone(), next_id);
            id_to_token.insert(next_id, merged.clone());
            merges.push(best.clone());
            next_id += 1;

            // Update word frequencies with merge
            let mut new_word_freqs = HashMap::new();
            for (word, freq) in word_freqs {
                let new_word = Self::apply_merge(&word, &best, &merged);
                *new_word_freqs.entry(new_word).or_insert(0) += freq;
            }
            word_freqs = new_word_freqs;
        }

        BPETokenizer {
            token_to_id,
            id_to_token,
            merges,
            vocab_size: next_id,
        }
    }

    fn apply_merge(word: &[String], pair: &(String, String), merged: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < word.len() {
            if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                result.push(merged.to_string());
                i += 2;
            } else {
                result.push(word[i].clone());
                i += 1;
            }
        }
        result
    }

    fn tokenize_word(&self, word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        tokens.push("</w>".to_string());

        // Apply merges in order
        for (a, b) in &self.merges {
            let merged = format!("{}{}", a, b);
            tokens = Self::apply_merge(&tokens, &(a.clone(), b.clone()), &merged);
        }

        tokens
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write vocab
        writeln!(file, "# Vocabulary")?;
        for i in 0..self.vocab_size {
            if let Some(token) = self.id_to_token.get(&i) {
                writeln!(file, "{}\t{}", i, token)?;
            }
        }

        // Write merges
        writeln!(file, "# Merges")?;
        for (a, b) in &self.merges {
            writeln!(file, "{} {}", a, b)?;
        }

        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut merges = Vec::new();
        let mut in_merges = false;
        let mut max_id = 0;

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("# Merges") {
                in_merges = true;
                continue;
            }
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if in_merges {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    merges.push((parts[0].to_string(), parts[1].to_string()));
                }
            } else {
                let parts: Vec<&str> = line.splitn(2, '\t').collect();
                if parts.len() == 2 {
                    let id: usize = parts[0].parse().unwrap_or(0);
                    let token = parts[1].to_string();
                    token_to_id.insert(token.clone(), id);
                    id_to_token.insert(id, token);
                    max_id = max_id.max(id);
                }
            }
        }

        Ok(BPETokenizer {
            token_to_id,
            id_to_token,
            merges,
            vocab_size: max_id + 1,
        })
    }
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            let tokens = self.tokenize_word(word);
            for token in tokens {
                let id = self.token_to_id.get(&token).copied().unwrap_or(UNK_ID);
                result.push(id);
            }
        }

        result
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let mut result = String::new();

        for &id in tokens {
            if id == PAD_ID || id == BOS_ID {
                continue;
            }
            if id == EOS_ID {
                break;
            }

            if let Some(token) = self.id_to_token.get(&id) {
                if token.ends_with("</w>") {
                    result.push_str(&token[..token.len() - 4]);
                    result.push(' ');
                } else if token != "</w>" {
                    result.push_str(token);
                }
            }
        }

        result.trim_end().to_string()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Word-level tokenizer with fixed vocabulary
#[derive(Debug, Clone)]
pub struct WordTokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    vocab_size: usize,
}

impl WordTokenizer {
    /// Create from a list of words
    pub fn from_vocab(words: &[&str]) -> Self {
        let mut word_to_id = HashMap::new();
        let mut id_to_word = HashMap::new();

        // Add special tokens
        let special = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN];
        for (i, &tok) in special.iter().enumerate() {
            word_to_id.insert(tok.to_string(), i);
            id_to_word.insert(i, tok.to_string());
        }

        let mut next_id = 4;
        for &word in words {
            if !word_to_id.contains_key(word) {
                word_to_id.insert(word.to_string(), next_id);
                id_to_word.insert(next_id, word.to_string());
                next_id += 1;
            }
        }

        WordTokenizer {
            word_to_id,
            id_to_word,
            vocab_size: next_id,
        }
    }

    /// Build vocabulary from corpus with frequency cutoff
    pub fn from_corpus(corpus: &str, max_vocab: usize, min_freq: usize) -> Self {
        let mut word_freqs: HashMap<String, usize> = HashMap::new();

        for word in corpus.split_whitespace() {
            let word = word.to_lowercase();
            *word_freqs.entry(word).or_insert(0) += 1;
        }

        // Sort by frequency
        let mut sorted: Vec<_> = word_freqs.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let mut word_to_id = HashMap::new();
        let mut id_to_word = HashMap::new();

        // Add special tokens
        let special = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN];
        for (i, &tok) in special.iter().enumerate() {
            word_to_id.insert(tok.to_string(), i);
            id_to_word.insert(i, tok.to_string());
        }

        let mut next_id = 4;
        for (word, freq) in sorted {
            if next_id >= max_vocab {
                break;
            }
            if freq < min_freq {
                break;
            }
            word_to_id.insert(word.clone(), next_id);
            id_to_word.insert(next_id, word);
            next_id += 1;
        }

        WordTokenizer {
            word_to_id,
            id_to_word,
            vocab_size: next_id,
        }
    }
}

impl Tokenizer for WordTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let word = word.to_lowercase();
                *self.word_to_id.get(&word).unwrap_or(&UNK_ID)
            })
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&id| {
                if id == PAD_ID || id == BOS_ID {
                    None
                } else if id == EOS_ID {
                    None
                } else {
                    self.id_to_word.get(&id).map(|s| s.as_str())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer() {
        let tokenizer = CharTokenizer::new();

        let text = "Hello, World!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_char_tokenizer_special() {
        let tokenizer = CharTokenizer::new();

        let mut tokens = tokenizer.encode("test");
        tokenizer.add_bos(&mut tokens);
        tokenizer.add_eos(&mut tokens);

        assert_eq!(tokens[0], BOS_ID);
        assert_eq!(tokens[tokens.len() - 1], EOS_ID);
    }

    #[test]
    fn test_bpe_tokenizer() {
        let corpus = "low lower lowest low lower lowest";
        let tokenizer = BPETokenizer::train(corpus, 50, 1);

        let tokens = tokenizer.encode("low");
        assert!(!tokens.is_empty());

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, "low");
    }

    #[test]
    fn test_word_tokenizer() {
        let corpus = "hello world hello there world";
        let tokenizer = WordTokenizer::from_corpus(corpus, 100, 1);

        let tokens = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_unknown_tokens() {
        let tokenizer = WordTokenizer::from_vocab(&["hello", "world"]);

        let tokens = tokenizer.encode("hello unknown world");

        // "unknown" should map to UNK_ID
        assert!(tokens.contains(&UNK_ID));
    }
}

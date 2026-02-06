use logos::Logos;
use std::hash::{Hash, Hasher};

/// Wrapper for f64 that implements Eq and Hash using bit representation
#[derive(Debug, Clone, Copy)]
pub struct Float(pub f64);

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for Float {
    fn from(f: f64) -> Self {
        Float(f)
    }
}

#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\n\f]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
pub enum Token {
    // Temporal operators
    #[token("fby")]
    Fby,

    #[token("next")]
    Next,

    #[token("prev")]
    Prev,

    #[token("first")]
    First,

    // Keywords
    #[token("where")]
    Where,

    #[token("if")]
    If,

    #[token("then")]
    Then,

    #[token("else")]
    Else,

    #[token("let")]
    Let,

    #[token("in")]
    In,

    #[token("fn")]
    Fn,

    // Training keywords
    #[token("param")]
    Param,

    #[token("train")]
    Train,

    #[token("model")]
    Model,

    #[token("loss")]
    Loss,

    #[token("optimizer")]
    OptimizerKw,

    #[token("epochs")]
    Epochs,

    #[token("batch_size")]
    BatchSize,

    #[token("lr")]
    Lr,

    #[token("input")]
    Input,

    #[token("target")]
    Target,

    // Initializers
    #[token("xavier")]
    Xavier,

    #[token("he")]
    He,

    #[token("zeros")]
    Zeros,

    #[token("ones")]
    Ones,

    // Optimizer types
    #[token("adam")]
    Adam,

    #[token("sgd")]
    Sgd,

    // Loss functions
    #[token("cross_entropy")]
    CrossEntropy,

    #[token("mse")]
    Mse,

    // Activation/tensor functions
    #[token("relu")]
    Relu,

    #[token("softmax")]
    Softmax,

    #[token("layer_norm")]
    LayerNorm,

    // Operators
    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Mul,

    #[token("/")]
    Div,

    #[token("@")]
    At,

    #[token("=")]
    Eq,

    #[token("==")]
    EqEq,

    #[token("!=")]
    Neq,

    #[token("<")]
    Lt,

    #[token("<=")]
    Lte,

    #[token(">")]
    Gt,

    #[token(">=")]
    Gte,

    #[token("&&")]
    And,

    #[token("||")]
    Or,

    #[token("!")]
    Not,

    // Delimiters
    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token(",")]
    Comma,

    #[token(";")]
    Semi,

    #[token("->")]
    Arrow,

    #[token(".")]
    Dot,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token(":")]
    Colon,

    // Literals and identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    #[regex(r"-?[0-9]+", |lex| lex.slice().parse().ok())]
    Integer(i64),

    #[regex(r"-?[0-9]+\.[0-9]+", |lex| lex.slice().parse().ok().map(Float))]
    Float(Float),

    #[token("true", |_| true)]
    #[token("false", |_| false)]
    Bool(bool),
}

pub fn tokenize(input: &str) -> Vec<Token> {
    Token::lexer(input)
        .filter_map(|result| result.ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_operators() {
        let tokens = tokenize("fib = 0 fby 1 fby fib + next fib");
        assert!(tokens.contains(&Token::Fby));
        assert!(tokens.contains(&Token::Next));
    }

    #[test]
    fn test_identifiers_and_numbers() {
        let tokens = tokenize("x = 42 + 3.14");
        assert!(matches!(&tokens[0], Token::Ident(s) if s == "x"));
        assert!(tokens.contains(&Token::Integer(42)));
        assert!(tokens.contains(&Token::Float(Float(3.14))));
    }

    #[test]
    fn test_dimension_syntax() {
        // Test dimension annotation: fby.t, next.batch, etc.
        let tokens = tokenize("1 fby.t 2");
        assert!(tokens.contains(&Token::Fby));
        assert!(tokens.contains(&Token::Dot));
        assert!(tokens.iter().any(|t| matches!(t, Token::Ident(s) if s == "t")));

        // Test bracket syntax for tensor literals: [1, 2, 3]
        let tokens = tokenize("[1, 2, 3]");
        assert!(tokens.contains(&Token::LBracket));
        assert!(tokens.contains(&Token::RBracket));
    }
}

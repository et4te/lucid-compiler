use crate::ast::*;
use crate::lexer::{Float, Token};
use chumsky::prelude::*;

pub fn parser() -> impl Parser<Token, Program, Error = Simple<Token>> {
    let ident = select! {
        Token::Ident(s) => s,
    };

    let integer = select! {
        Token::Integer(n) => Expr::Int(n),
    };

    let float = select! {
        Token::Float(Float(f)) => Expr::Float(f),
    };

    let boolean = select! {
        Token::Bool(b) => Expr::Bool(b),
    };

    // Parse optional dimension annotation: .dim
    let dim_annotation = just(Token::Dot)
        .ignore_then(select! { Token::Ident(s) => s })
        .map(|s| Some(Dimension::new(s)))
        .or_not()
        .map(|opt| opt.flatten());

    let expr = recursive(|expr| {
        // Box the recursive expr to reduce type depth
        let expr_boxed = expr.clone().boxed();

        // Tensor literal: [expr, expr, ...]
        let tensor = expr_boxed
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(Expr::Tensor);

        let atom = integer
            .or(float)
            .or(boolean)
            .or(tensor)
            .or(ident.clone().map(Expr::Var))
            .or(expr_boxed
                .clone()
                .delimited_by(just(Token::LParen), just(Token::RParen)))
            .boxed();

        // Lambda expressions: fn(x, y) -> expr
        let lambda = just(Token::Fn)
            .ignore_then(
                ident
                    .clone()
                    .separated_by(just(Token::Comma))
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .then_ignore(just(Token::Arrow))
            .then(expr_boxed.clone())
            .map(|(params, body)| Expr::Lambda {
                params,
                body: Box::new(body),
            });

        // Unary operators
        let unary = choice((
            just(Token::Minus).to(UnOp::Neg),
            just(Token::Not).to(UnOp::Not),
        ))
        .repeated()
        .then(atom.or(lambda))
        .foldr(|op, expr| Expr::UnOp {
            op,
            expr: Box::new(expr),
        })
        .boxed();

        // Index operation: expr[expr, expr, ...]
        let indexed = unary.clone().then(
            expr_boxed
                .clone()
                .separated_by(just(Token::Comma))
                .delimited_by(just(Token::LBracket), just(Token::RBracket))
                .repeated(),
        ).foldl(|tensor, indices| Expr::Index {
            tensor: Box::new(tensor),
            indices,
        });

        // Activation functions: relu(expr), softmax(expr), softmax.dim(expr), layer_norm(expr)
        let relu_call = just(Token::Relu)
            .ignore_then(expr_boxed.clone().delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|e| Expr::Relu(Box::new(e)));

        let softmax_call = just(Token::Softmax)
            .ignore_then(dim_annotation.clone())
            .then(expr_boxed.clone().delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|(dim, e)| Expr::Softmax {
                dim,
                expr: Box::new(e),
            });

        let layer_norm_call = just(Token::LayerNorm)
            .ignore_then(dim_annotation.clone())
            .then(expr_boxed.clone().delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|(dim, e)| Expr::LayerNorm {
                dim,
                expr: Box::new(e),
            });

        let builtin_funcs = choice((relu_call, softmax_call, layer_norm_call));

        // Function application
        let call = indexed
            .or(builtin_funcs)
            .then(
                expr_boxed
                    .clone()
                    .separated_by(just(Token::Comma))
                    .delimited_by(just(Token::LParen), just(Token::RParen))
                    .repeated(),
            )
            .foldl(|func, args| Expr::App {
                func: Box::new(func),
                args,
            })
            .boxed();

        // Matrix multiplication with optional dimension: @ or @.dim
        let matmul = call
            .clone()
            .then(
                just(Token::At)
                    .ignore_then(dim_annotation.clone())
                    .then(call.clone())
                    .repeated(),
            )
            .foldl(|left, (dim, right)| Expr::MatMul {
                dim,
                left: Box::new(left),
                right: Box::new(right),
            })
            .boxed();

        // Binary operators with precedence
        let product = matmul
            .clone()
            .then(
                choice((
                    just(Token::Mul).to(BinOp::Mul),
                    just(Token::Div).to(BinOp::Div),
                ))
                .then(matmul.clone())
                .repeated(),
            )
            .foldl(|left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
            .boxed();

        let sum = product
            .clone()
            .then(
                choice((
                    just(Token::Plus).to(BinOp::Add),
                    just(Token::Minus).to(BinOp::Sub),
                ))
                .then(product)
                .repeated(),
            )
            .foldl(|left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
            .boxed();

        let comparison = sum
            .clone()
            .then(
                choice((
                    just(Token::EqEq).to(BinOp::Eq),
                    just(Token::Neq).to(BinOp::Neq),
                    just(Token::Lt).to(BinOp::Lt),
                    just(Token::Lte).to(BinOp::Lte),
                    just(Token::Gt).to(BinOp::Gt),
                    just(Token::Gte).to(BinOp::Gte),
                ))
                .then(sum)
                .repeated(),
            )
            .foldl(|left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
            .boxed();

        let logical = comparison
            .clone()
            .then(
                choice((
                    just(Token::And).to(BinOp::And),
                    just(Token::Or).to(BinOp::Or),
                ))
                .then(comparison)
                .repeated(),
            )
            .foldl(|left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
            .boxed();

        // Fby operator with optional dimension: fby or fby.dim
        let fby_op = just(Token::Fby)
            .ignore_then(dim_annotation.clone());

        let fby = logical
            .clone()
            .then(fby_op.then(expr_boxed.clone()).repeated())
            .map(|(init, nexts)| {
                nexts.into_iter().fold(init, |init, (dim, next)| Expr::Fby {
                    dim,
                    init: Box::new(init),
                    next: Box::new(next),
                })
            })
            .boxed();

        // Temporal operators with dimension support
        let temporal_ops = choice((
            just(Token::Next)
                .ignore_then(dim_annotation.clone())
                .then(fby.clone())
                .map(|(dim, e)| Expr::Next {
                    dim,
                    expr: Box::new(e),
                }),
            just(Token::Prev)
                .ignore_then(dim_annotation.clone())
                .then(fby.clone())
                .map(|(dim, e)| Expr::Prev {
                    dim,
                    expr: Box::new(e),
                }),
            just(Token::First)
                .ignore_then(dim_annotation.clone())
                .then(fby.clone())
                .map(|(dim, e)| Expr::First {
                    dim,
                    expr: Box::new(e),
                }),
            fby,
        ))
        .boxed();

        // If-then-else
        let if_expr = just(Token::If)
            .ignore_then(temporal_ops.clone())
            .then_ignore(just(Token::Then))
            .then(temporal_ops.clone())
            .then_ignore(just(Token::Else))
            .then(temporal_ops.clone())
            .map(|((cond, then_branch), else_branch)| Expr::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            });

        // Let expressions
        let binding = ident.clone().then_ignore(just(Token::Eq)).then(expr_boxed.clone());

        let let_expr = just(Token::Let)
            .ignore_then(binding.clone().separated_by(just(Token::Comma)))
            .then_ignore(just(Token::In))
            .then(expr_boxed.clone())
            .map(|(bindings, body)| Expr::Let {
                bindings,
                body: Box::new(body),
            });

        // Where expressions
        let where_expr = temporal_ops
            .clone()
            .then_ignore(just(Token::Where))
            .then(
                binding
                    .separated_by(just(Token::Comma))
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(|(expr, bindings)| Expr::Where {
                expr: Box::new(expr),
                bindings,
            });

        choice((if_expr, let_expr, where_expr, temporal_ops))
    });

    // Initializer: xavier, he, zeros, ones
    let initializer = choice((
        just(Token::Xavier).to(Initializer::Xavier),
        just(Token::He).to(Initializer::He),
        just(Token::Zeros).to(Initializer::Zeros),
        just(Token::Ones).to(Initializer::Ones),
    ));

    // Shape: [dim1, dim2, ...]
    let shape_spec = expr
        .clone()
        .separated_by(just(Token::Comma))
        .delimited_by(just(Token::LBracket), just(Token::RBracket));

    // Parameter declaration: param name: [shape] = initializer;
    let param_decl = just(Token::Param)
        .ignore_then(ident.clone())
        .then_ignore(just(Token::Colon))
        .then(shape_spec.clone())
        .then_ignore(just(Token::Eq))
        .then(initializer)
        .then_ignore(just(Token::Semi))
        .map(|((name, shape), init)| ParamDecl {
            name,
            shape,
            initializer: init,
        });

    // Optimizer: adam(lr=0.001) or sgd(lr=0.01)
    let optimizer_spec = choice((
        just(Token::Adam)
            .ignore_then(
                just(Token::Lr)
                    .ignore_then(just(Token::Eq))
                    .ignore_then(select! { Token::Float(Float(f)) => f })
                    .delimited_by(just(Token::LParen), just(Token::RParen))
            )
            .map(|lr| OptimizerType { kind: OptimizerKind::Adam, learning_rate: lr }),
        just(Token::Sgd)
            .ignore_then(
                just(Token::Lr)
                    .ignore_then(just(Token::Eq))
                    .ignore_then(select! { Token::Float(Float(f)) => f })
                    .delimited_by(just(Token::LParen), just(Token::RParen))
            )
            .map(|lr| OptimizerType { kind: OptimizerKind::Sgd, learning_rate: lr }),
    ));

    // Loss function: cross_entropy or mse
    let loss_spec = choice((
        just(Token::CrossEntropy).to(LossFunction::CrossEntropy),
        just(Token::Mse).to(LossFunction::Mse),
    ));

    // Integer value for epochs/batch_size
    let int_val = select! { Token::Integer(n) => n as usize };

    // Train block: train { input: x, target: y, model: expr, loss: cross_entropy, optimizer: adam(lr=0.001), epochs: 10, batch_size: 32 }
    let train_block = just(Token::Train)
        .ignore_then(just(Token::LBrace))
        .ignore_then(
            just(Token::Input).ignore_then(just(Token::Colon)).ignore_then(ident.clone())
        )
        .then_ignore(just(Token::Comma))
        .then(
            just(Token::Target).ignore_then(just(Token::Colon)).ignore_then(ident.clone())
        )
        .then_ignore(just(Token::Comma))
        .then(
            just(Token::Model).ignore_then(just(Token::Colon)).ignore_then(expr.clone())
        )
        .then_ignore(just(Token::Comma))
        .then(
            just(Token::Loss).ignore_then(just(Token::Colon)).ignore_then(loss_spec)
        )
        .then_ignore(just(Token::Comma))
        .then(
            just(Token::OptimizerKw).ignore_then(just(Token::Colon)).ignore_then(optimizer_spec)
        )
        .then_ignore(just(Token::Comma))
        .then(
            just(Token::Epochs).ignore_then(just(Token::Colon)).ignore_then(int_val.clone())
        )
        .then_ignore(just(Token::Comma))
        .then(
            just(Token::BatchSize).ignore_then(just(Token::Colon)).ignore_then(int_val)
        )
        .then_ignore(just(Token::RBrace))
        .map(|((((((input, target), model), loss), optimizer), epochs), batch_size)| {
            TrainConfig {
                input,
                target,
                model,
                loss,
                optimizer: OptimizerType {
                    kind: optimizer.kind,
                    learning_rate: optimizer.learning_rate,
                },
                epochs,
                batch_size,
                learning_rate: optimizer.learning_rate,
            }
        });

    let definition = choice((
        just(Token::Fn)
            .ignore_then(ident.clone())
            .then(
                ident
                    .clone()
                    .separated_by(just(Token::Comma))
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then_ignore(just(Token::Semi))
            .map(|((name, params), body)| Definition::Function {
                name,
                params,
                body,
            }),
        ident
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then_ignore(just(Token::Semi))
            .map(|(name, expr)| Definition::Equation { name, expr }),
    ));

    // Parse: params, then definitions, then optional train block, then main expression
    param_decl
        .repeated()
        .then(definition.repeated())
        .then(train_block.or_not())
        .then(expr)
        .map(|(((params, definitions), train_config), main_expr)| Program {
            definitions,
            params,
            train_config,
            main_expr,
        })
        .then_ignore(end())
}

pub fn parse(tokens: Vec<Token>) -> Result<Program, Vec<Simple<Token>>> {
    parser().parse(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    #[test]
    fn test_simple_expr() {
        let tokens = tokenize("1 + 2");
        let result = parse(tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fby() {
        let tokens = tokenize("0 fby 1");
        let result = parse(tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fby_with_dimension() {
        let tokens = tokenize("0 fby.t 1");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::Fby { dim, .. } => {
                    assert!(dim.is_some());
                    assert_eq!(dim.as_ref().unwrap().0, "t");
                }
                _ => panic!("Expected Fby expression"),
            }
        }
    }

    #[test]
    fn test_tensor_literal() {
        let tokens = tokenize("[1, 2, 3]");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::Tensor(elements) => {
                    assert_eq!(elements.len(), 3);
                }
                _ => panic!("Expected Tensor expression"),
            }
        }
    }

    #[test]
    fn test_nested_tensor() {
        let tokens = tokenize("[[1, 2], [3, 4]]");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::Tensor(rows) => {
                    assert_eq!(rows.len(), 2);
                    match &rows[0] {
                        Expr::Tensor(cols) => assert_eq!(cols.len(), 2),
                        _ => panic!("Expected nested Tensor"),
                    }
                }
                _ => panic!("Expected Tensor expression"),
            }
        }
    }

    #[test]
    fn test_next_with_dimension() {
        let tokens = tokenize("next.seq x");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::Next { dim, .. } => {
                    assert!(dim.is_some());
                    assert_eq!(dim.as_ref().unwrap().0, "seq");
                }
                _ => panic!("Expected Next expression"),
            }
        }
    }

    #[test]
    fn test_multi_dimensional_fby() {
        // Sequence in time dimension, then in batch dimension
        let tokens = tokenize("(1 fby.t 2) fby.batch 3");
        let result = parse(tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_matmul_operator() {
        // Basic matrix multiplication
        let tokens = tokenize("a @ b");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::MatMul { dim, .. } => {
                    assert!(dim.is_none());
                }
                _ => panic!("Expected MatMul expression"),
            }
        }
    }

    #[test]
    fn test_matmul_with_dimension() {
        // Batched matrix multiplication
        let tokens = tokenize("x @.batch weights");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::MatMul { dim, .. } => {
                    assert!(dim.is_some());
                    assert_eq!(dim.as_ref().unwrap().0, "batch");
                }
                _ => panic!("Expected MatMul expression"),
            }
        }
    }

    #[test]
    fn test_relu_call() {
        let tokens = tokenize("relu(x)");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::Relu(_) => {}
                _ => panic!("Expected Relu expression"),
            }
        }
    }

    #[test]
    fn test_softmax_with_dimension() {
        let tokens = tokenize("softmax.seq(x)");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::Softmax { dim, .. } => {
                    assert!(dim.is_some());
                    assert_eq!(dim.as_ref().unwrap().0, "seq");
                }
                _ => panic!("Expected Softmax expression"),
            }
        }
    }

    #[test]
    fn test_layer_norm_with_dimension() {
        let tokens = tokenize("layer_norm.hidden(x)");
        let result = parse(tokens);
        assert!(result.is_ok());

        if let Ok(program) = result {
            match &program.main_expr {
                Expr::LayerNorm { dim, .. } => {
                    assert!(dim.is_some());
                    assert_eq!(dim.as_ref().unwrap().0, "hidden");
                }
                _ => panic!("Expected LayerNorm expression"),
            }
        }
    }

    #[test]
    fn test_chained_matmul() {
        // Multiple matrix multiplications
        let tokens = tokenize("a @.batch b @.batch c");
        let result = parse(tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_like_expr() {
        // Expression that looks like transformer attention
        let tokens = tokenize("softmax.seq(q @.batch k) @.batch v");
        let result = parse(tokens);
        assert!(result.is_ok());
    }
}

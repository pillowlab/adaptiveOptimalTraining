function v = sparse_logdet(A)
% computes log determinant of a symmetric & positive semi-definite matrix
% input matrix can be sparse as well as full

% using cholesky decomposition (symmetric & positive semi-definite)
v = 2 * full( sum(log(diag(chol(A)))) );

% note: reference: Dahua Lin's logdet function

end

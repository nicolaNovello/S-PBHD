function H = ParaHerm(G);
%H = ParaHerm(G)
%
%   Returns the parahermitian (i.e. the complex conjugate transpose, time
%   reversed matrix) of the MIMO system matrix G.
%  
%   If G represents a polynomial matrix G(z) of order L with
%      G(z) = G0 + G1 z^{-1} + G2 z^{-2} + ... + GL z^{-L}
%   then
%      G(:,:,1) = G0;
%      G(:,:,2) = G1;
%      G(:,:,3) = G2;
%      ...
%      G(:,:,L) = GL;
%   The parahermitian H(z) = G~(z) is given by
%      H(:,:,1)   = GL';
%      ...
%      H(:,:,L-1) = G1';
%      H(:,:,L)   = G0';
%   Note that (.)' is the Hermitian transpose operator.
%
%   Input parameter:
%      H      K x N x L    MIMO system matrix
%
%   Output parameter:
%      G      N x K x L    MIMO system matrix
 
% S Weiss, Univ of Southampton, 15/7/2004
  
[M,N,L] = size(G);
H = zeros(N,M,L);
for m = 1:M,
  for n = 1:N,
     H(n,m,:) = conj(G(m,n,end:-1:1));
  end;  
end;


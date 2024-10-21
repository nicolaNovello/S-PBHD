function Rt = ParaHermDFT(Rf);
%Rt = ParaHermIDFT(Rf,Nfft);
%
%  Rt = ParaHermIDFT(Rf) returns the IDFT of a parahermitian matrix stored in
%  Rf, with indices running from DC to 2pi. The matrix Rf is of dimension MxMxL,
%  where M is the spatial dimension, and L the number of sample points on the
%  unit circle.
%
%  The function returns an odd-length time-domain version, whereby the length
%  is appropriate trimmed in case the time domain support is shorter than the 
%  number of sample points.
%
%  Input parameters:
%    Rf         cross-spectral density matrix
% 
%  Output parameter:
%    Rt         space-time covariance-type matrix

% S. Weiss, 18/4/2021

[M,~,L] = size(Rf);
Threshold = 10^(-14);

% apply inverse DFT
R_td = ifft(Rf,L,3);

% rearrange time domain data
if mod(L,2)==0,             % even length
  dummy = zeros(M,M,L+1);
  dummy(:,:,L/2+1:L+1) = R_td(:,:,1:L/2+1);
  dummy(:,:,1:L/2) = R_td(:,:,L/2+1:L);
else                        % odd length
  dummy = zeros(M,M,L);
  dummy(:,:,(L+1)/2:L) = R_td(:,:,1:(L+1)/2);
  dummy(:,:,1:(L-1)/2) = R_td(:,:,(L+1)/2+1:L);
end;
  
% trim leading and trailing zeros
L = size(dummy,3);
Power = zeros((L+1)/2,1);
for l = 1:(L+1)/2,
   Power(l) = norm(dummy(:,:,l),'fro');
end;
Index = max(find(cumsum(Power)<Threshold));
Rt = dummy(:,:,Index+1:L-Index);

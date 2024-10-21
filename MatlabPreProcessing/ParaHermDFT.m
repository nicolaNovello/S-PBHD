function Rf = ParaHermDFT(Rt,Nfft);
%Rf = ParaHermDFT(Rt,Nfft);
%
%  Rf = ParaHermDFT(Rt,Nfft) returns the Nfft-point DFT of a space-time
%  covariance matrix stored in Rt. The matrix Rt is of dimension MxMxL,
%  where M is the spatial dimension, and L the lag dimension (assumed to be odd).
%
%  The function first aligns Rt, such that the zero-lag component is indeed
%  at zero.
%
%  Nfft must be larger than L.
%
%  Input parameters:
%    Rt         space-time covariance matrix
%    Nfft       DFT length
% 
%  Output parameter:
%    Rf         MxMxNfft cross-spectral density matrix

% S. Weiss, 9/3/2018

[M,~,L] = size(Rt);
if L > Nfft,
   error('DFT length too short');
end;

% rearrange time domain data
L2 = (L+1)/2;
R_td = zeros(M,M,Nfft);
R_td(:,:,1:L2) = Rt(:,:,L2:L);
R_td(:,:,Nfft-L2+2:Nfft) = Rt(:,:,1:L2-1);

% apply DFT
Rf = fft(R_td,Nfft,3);


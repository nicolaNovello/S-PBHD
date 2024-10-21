% By Frank Ebong, Department of Networked and Embedded Systems, University of Klagenfurt, December 2023. 

num=320;

file1=""+(1:num)+".s4p";

file="60.s4p";
VNA=sparameters(file);
freq=VNA.Frequencies;
numfreq=numel(freq);

for m1=1:320
 Hf{m1}=zeros(4,4,1000);
 % H{m1}=zeros(4,4,1000);
 % R{m1}=zeros(4,4,1000);
end

for m1=1:320
 H{m1}=zeros(4,4,1000);
 % R{m1}=zeros(4,4,1000);
end

for m1=1:320
 %Hf0{m1}=zeros(4,4,1000);
 % H{m1}=zeros(4,4,1000);
 R{m1}=zeros(4,4,1000);
end



Hf= zeros(4,4,numfreq); 

for n=1:320
VNA2=sparameters(file1(n));
for i=1:4
    for j=1:4
        if i==j;
        else
         Hf(i,j,:)=rfparam(VNA,i,j);
    end
  end 

end 
H0{1,n}=Hf(:,:,:);
end 

for n2=1:320
  H{1,n2} = ifft(H0{1,n2},2000,3);
  R{1,n2}= PolyMatConv(H{1,n2},ParaHerm(H{1,n2}));
  Rf{1,n2} = ParaHermDFT(R{1,n2},4096);
end


% H = ifft(Hf0,2000,3);
% R = PolyMatConv(H,ParaHerm(H));
% 
% % take bin-wise EVD
% Rf = ParaHermDFT(R,4096);
% EigVals = zeros(4,4096);
for n3=1:320
 for k =1:4096,
  [~,EigVals{1,n3}(:,k)] = eig(Rf{1,n3}(:,:,k),'vector');
 end
end

Lambda1=zeros(4096,320);
Lambda2=zeros(4096,320);
Lambda3=zeros(4096,320);
Lambda4=zeros(4096,320);

for n4=1:320
  Lambda1(:,n4)=10*log10(abs(EigVals{1,n4}(1,:)));  
  Lambda2(:,n4)=10*log10(abs(EigVals{1,n4}(2,:)));
  Lambda3(:,n4)=10*log10(abs(EigVals{1,n4}(3,:)));
  Lambda4(:,n4)=10*log10(abs(EigVals{1,n4}(4,:)));

end

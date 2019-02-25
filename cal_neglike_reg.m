function [neglike] = cal_neglike_reg(true_y, pred_y, coef,n)

e = bsxfun(@minus, pred_y, true_y);
k = sum(coef ~= 0);
s = sum(e.^2)/(n-k);

neglogprob = bsxfun(@plus, bsxfun(@rdivide,e.^2,s*2), (1/2)*log(2*pi*s));
neglike = sum(neglogprob,1);

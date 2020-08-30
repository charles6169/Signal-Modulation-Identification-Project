function   y_cum = myCUM4EST (y, maxlag, nsamp, overlap, flag, k1, k2)
%CUM4EST Fourth-order cumulants.
%       Should be invoked via CUMEST for proper parameter checks
%       y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)

%       Computes sample estimates of fourth-order cumulants
%       via the overlapped segment method.
%
%       y_cum = cum4est (y, maxlag, samp_seg, overlap, flag, k1, k2)
%              y: input data vector (column)
%         maxlag: maximum lag
%       samp_seg: samples per segment
%        overlap: percentage overlap of segments
%          flag : 'biased', biased estimates are computed
%               : 'unbiased', unbiased estimates are computed.
%	  k1,k2 : the fixed lags in C3(m,k1) or C4(m,k1,k2); see below
%	  y_cum : estimated fourth-order cumulant slice
%	          C4(m,k1,k2)  -maxlag <= m <= maxlag
% 	Note: all parameters must be specified


%  Copyright (c) 1991-2001 by United Signals & Systems, Inc. 
%       $Revision: 1.4 $
%  A. Swami   January 20, 1993

% Modified, Januar 20, 1994 to handle complex case properly:
% c4(t1,t2,t3) := cum( x^*(t), x(t+t1), x(t+t2), x^*(t+t3) )

%     RESTRICTED RIGHTS LEGEND
% Use, duplication, or disclosure by the Government is subject to
% restrictions as set forth in subparagraph (c) (1) (ii) of the
% Rights in Technical Data and Computer Software clause of DFARS
% 252.227-7013.
% Manufacturer: United Signals & Systems, Inc., P.O. Box 2374,
% Culver City, California 90231.
%
%  This material may be reproduced by or for the U.S. Government pursuant
%  to the copyright license under the clause at DFARS 252.227-7013.


% c4(t1,t2,t3) := cum( x^*(t), x(t+t1), x(t+t2), x^*(t+t3) )
%  cum(w,x,y,z) := E(wxyz) - E(wx)E(yz) - E(wy)E(xz) - E(wz)E(xy)
%  and, w,x,y,z are assumed to be zero-mean.


% ---- Parameter checks are done in CUMEST ----------------------
   [n1,n2]  = size(y);
   N        = n1 * n2;
   overlap0 = overlap;
   overlap  = fix(overlap/100 * nsamp);
   nrecord  = fix( (N - overlap)/(nsamp - overlap) );
   nadvance = nsamp - overlap;


% ------ scale factors for unbiased estimates --------------------

   nlags = 2 * maxlag + 1;
   zlag  = 1 + maxlag;
   tmp   = zeros(nlags,1);
   if (flag(1:1) == 'b'  | flag(1:1) == 'B')
       scale = ones(nlags,1) / nsamp;
   else
       ind   = [-maxlag:maxlag]';
       kmin  = min(0,min(k1,k2));
       kmax  = max(0,max(k1,k2));
       scale = nsamp - max(ind,kmax) + min(ind,kmin);
       scale = ones(nlags,1) ./ scale;
   end
   mlag  = maxlag + max(abs([k1,k2]));
   mlag  = max( mlag, abs(k1-k2) );
   mlag1 = mlag + 1;
   nlag  = maxlag;
   m2k2  = zeros(2*maxlag+1,1);

   if (any(any(imag(y) ~= 0))) complex_flag = 1;
   else complex_flag = 0;
   end

% ----------- estimate second- and fourth-order moments; combine ------

   y_cum  = zeros(2*maxlag+1,1);
   R_yy   = zeros(2*mlag+1,1);

   ind   = 1:nsamp;
   for i=1:nrecord
       tmp = y_cum * 0 ;
       x = y(ind); x = x(:) - mean(x);  z =  x * 0;  cx = conj(x);
%                     create the "IV" matrix: offset for second lag

       if (k1 >= 0)
       		z(1:nsamp-k1)  = x(1:nsamp-k1,:) .* cx(k1+1: nsamp,:);
       else
       		z(-k1+1:nsamp) = x(-k1+1:nsamp)  .* cx(1:nsamp+k1);
       end

%                     create the "IV" matrix: offset for third lag

       if (k2 >= 0)
          z(1:nsamp-k2) = z(1:nsamp-k2) .* x(k2+1: nsamp);
          z(nsamp-k2+1:nsamp) = zeros(k2,1);
       else
          z(-k2+1:nsamp) = z(-k2+1:nsamp) .* x(1:nsamp+k2);
          z(1:-k2)    = zeros(-k2,1);
       end

       tmp(zlag)  =  tmp(zlag) + z' * x;
       for k = 1:maxlag
           tmp(zlag-k) = tmp(zlag-k) + z([k+1:nsamp])' * x([1:nsamp-k]);
           tmp(zlag+k) = tmp(zlag+k) + z([1:nsamp-k])' * x([k+1:nsamp]);
       end

       y_cum = y_cum + tmp .* scale ;

       R_yy = cum2est(x,mlag,nsamp,overlap0,flag);
       if (complex_flag)    % We need E x(t)x(t+tau) stuff also:
       	   M_yy  = cum2x(conj(x),x,mlag,nsamp,overlap0,flag);
       else
           M_yy  = R_yy;
       end
       y_cum = y_cum ...
           - R_yy(mlag1+k1) * R_yy(mlag1-k2-nlag:mlag1-k2+nlag) ...
           - R_yy(k1-k2+mlag1) * R_yy(mlag1-nlag:mlag1+nlag)  ...
           - M_yy(mlag1+k2)' * M_yy(mlag1-k1-nlag:mlag1-k1+nlag) ;

       ind = ind + nadvance;
end

y_cum = y_cum / nrecord;
return

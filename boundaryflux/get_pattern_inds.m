function [idf, idt, iuf, iut, ilf, ilt, irf, irt] = get_pattern_inds(pattern)
%% Get illumination pattern

% Pattern can be generated by:
% pattern = roipoly(N,N);

%% Now you have an illumination pattern, find 2D boundaries with flux left, right, up, down
inds_down  = find(conv2(pattern, [1;0;-1], 'same') > 0);
inds_up    = find(conv2(pattern, [1;0;-1], 'same') < 0);
inds_right = find(conv2(pattern, [1 0 -1], 'same') > 0);
inds_left  = find(conv2(pattern, [1 0 -1], 'same') < 0);

%% gets the indices of pattern boundaries for each direction of flux
sz = size(pattern);

[id,jd]   = ind2sub(sz, inds_down);
[iu,ju]   = ind2sub(sz, inds_up);
[ir,jr]   = ind2sub(sz, inds_right);
[il,jl]   = ind2sub(sz, inds_left);

idt = inds_down;
idf = sub2ind(sz, id-1,jd);
iut = inds_up;
iuf = sub2ind(sz, iu+1,ju);
ilt = inds_left;
ilf = sub2ind(sz, il,jl+1);
irt = inds_right;
irf = sub2ind(sz, ir,jr-1);


%         % Flux upwards:
%         dydt(iu+1,ju) = dydt(iu+1,ju) - f*y(iu+1,ju); % flux FROM
%         dydt(iu,ju)   = dydt(iu,ju)   + f*y(iu+1,ju); % flux TO
%         % Flux downwards:
%         dydt(id-1,jd) = dydt(id-1,jd) + f*y(id-1,jd); % flux FROM
%         dydt(id,jd)   = dydt(id,jd)   + f*y(id-1,jd); % flux TO
%         % Flux leftwards:
%         dydt(il,jl+1) = dydt(il,jl+1) - f*y(il,jl+1); % flux FROM
%         dydt(il,jl)   = dydt(il,jl)   + f*y(il,jl+1); % flux TO
%         % Flux rightwards:
%         dydt(ir,jr-1) = dydt(ir,jr-1) - f*y(ir,jr-1); % flux FROM
%         dydt(ir,jr)   = dydt(ir,jr)   + f*y(ir,jr-1); % flux TO
return

%% sanity check: can plot the results
tmp = zeros(N,N);
for k = 1:length(idown)
    tmp(idown(k), jdown(k)) = 1;
end
for k = 1:length(iup)
    tmp(iup(k), jup(k)) = 2;
end
for k = 1:length(iright)
    tmp(iright(k), jright(k)) = 3;
end
for k = 1:length(ileft)
    tmp(ileft(k), jleft(k)) = 4;
end
imagesc(tmp), colorbar

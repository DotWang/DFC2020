function X_result = drawresult(labels,row,col)
num_class = 10;

palette = [[  0., 153.,   0.],
       [198., 176.,  68.],
       [251., 255.,  19.],
       [182., 255.,   5.],
       [ 39., 255., 135.],
       [194.,  79.,  68.],
       [165., 165., 165.],
       [105., 255., 248.],
       [249., 255., 164.],
       [ 28.,  13., 255.]]/255.;

X_result = zeros(size(labels,1),3);
for i=1:num_class
    X_result(find(labels==i),1) = palette(i,1);
    X_result(find(labels==i),2) = palette(i,2);
    X_result(find(labels==i),3) = palette(i,3);
end

X_result = reshape(X_result,row,col,3);
end
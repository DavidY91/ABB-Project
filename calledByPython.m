function Transform = calledByPython(X, Y)
X = from_list_to_mat(X);
Y = from_list_to_mat(Y);
opt.method='nonrigid';
opt.normalize = 1;
opt.corresp = true;
opt.outliers = 0.1;
opt.viz = 0;
[Transform, C]=cpd_register(X, Y, opt);
Transform.C = uint16(C)';
% figure,cpd_plot_iter(X, Y); title('Before');
% figure,cpd_plot_iter(X, Transform.Y);  title('After registering Y to X');
end

function mat = from_list_to_mat(list)
mat = zeros(size(list,2),2);
for i = 1:size(list,2)
    tempx = list{i}(1);
    tempy = list{i}(2);
    mat(i,:) = [tempx{1}, tempy{1}];
end
end

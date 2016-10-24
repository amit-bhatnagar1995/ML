function plotData(X, y, pausetime)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

num5= find(y==5); 
num50 = find(y == 50);
num500= find(y==500); 
num5000 = find(y == 5000);
num50000= find(y==50000); 
num500000 = find(y == 500000);

y=log(y);

for i= 1:size(X, 2)
    
    plot(X(num5, i), y(num5, 1), 'r+','LineWidth', 2, ...
    'MarkerSize', 7);
    hold on;
    plot(X(num50, i), y(num50, 1), 'bo', 'MarkerFaceColor', 'y', ...
    'MarkerSize', 7);
    hold on;
    plot(X(num500, i), y(num500, 1), 'gx','LineWidth', 2, ...
    'MarkerSize', 7);
    hold on;
    plot(X(num5000, i), y(num5000, 1), 'c*', 'MarkerFaceColor', 'y', ...
    'MarkerSize', 7);
    hold on;
    plot(X(num50000, i), y(num50000, 1), 'm.','LineWidth', 2, ...
    'MarkerSize', 7);
    hold on;
    plot(X(num500000, i), y(num500000, 1), 'k^','LineWidth', 2, ...
    'MarkerSize', 7);
    legend('5', '50', '500', '5000', '50000', '500000');
    xlabel(['feature ', num2str(i-1)]);
    
    pause(pausetime);
    %close all;
end
% =========================================================================



hold off;

end

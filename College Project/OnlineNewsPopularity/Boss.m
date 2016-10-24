%% ======== Part 1: read the input from a file=============================
% read from a .csv file into variable "input"

filename='OnlineNewsPopularity.csv';
startrow = 1;
endrow = 39645;

% input varible may be initialised once and for the rest of the operations,
% one may just comment the line as the variable with the data has already
% been feeded into the variable workspace of the matlab
%
% ->input=import_the_data(filename, startrow, endrow);
% ->input(isfinite(input(:, 1)), :);
%

%% ======== Part 2: extract the relevant data for the training=============

% check for the dimensions of input, if need be
%    disp_the_size(input);
%
% storing the feature matrix in X and output in y vector 
[r c]=size(input);

% first row is in NaN, hence row 1 is removed
% column 1 has non predictive characteristic
a=floor((r-1)*0.8);
X=input(2:a, 2:c-1);    predict_X=  input(a+1:end, 2:c-1);
y=input(2:a, c);        predict_y=  input(a+1:end, c);
m=length(y);

%disp_the_size(X);
%disp_the_size(y);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ======= Part 3: Visualise the data ====================================

% plotting the values of shares against each of the feature vectors
% seperately.

pausetime=0.1;
% ->
plotData(X, y, pausetime);

fprintf('Program paused. Press enter to continue.\n');
pause;
close ;

%% ======= Part 4: Manipulate the Data =============================
% Feature normalisation

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X_norm mu sigma] = featureNormalize(X);

% Add intercept term to X
X_norm = [ones(m, 1) X_norm];


%% ======= Part 5: Gradient Descent ================================

% Choose some alpha value
alpha = 0.01;
% for OnlineNewsProperty this is a better choice than 0.01 and 0.03
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta_GD = zeros(size(X_norm, 2), 1);
[theta_GD, J_history] = gradientDescentMulti(X_norm, y, theta_GD, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
saveas(gcf, 'Cost Function with iterations.jpg');
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta_GD);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
close;

%% ======= Part 6: Prediction Using Gradient Descent =======================================

% normalise the features of predict_X using the mu and sigma obtained by
% the normalising of the training set formed by 70% of the "input"
%
%predict_X=input(2:a, 2:c-1);
%predict_y=input(2:a, c);
temp =predict_X;
for i=1:size(temp, 1)
    temp(i, :)=temp(i, :)-mu;
    temp(i, :)=temp(i, :)./sigma;
end
% add intercept term
temp=[ones(size(temp,1), 1) temp];

% predicting the values for predict_X using the theta obtained
% using Gradient Descent Method.
predict_GD = temp * theta_GD;

% Train Accuracy
fprintf('Train Accuracy Using Gradient Descent: %f\n', mean(double(predict_GD == predict_y)) * 100);

%num=0;
%for i=1: size(predict_y, 1)
%   pp=predict_y(i, 1);
%   qq=predict_GD(i, 1);
% 
%   if qq >= 0.8*pp && qq <= 1.2*pp
%       num= num+1;
%   end
%end
%disp('GDGDGD');
%disp(num/size(predict_y, 1));

% compare the predict_GD and predict_y

diff_GD=abs(predict_GD-predict_y);

figure;

plot( predict_y,predict_GD, 'color' , 'b');
hold on;
plot(0:max(predict_GD)+10, 0:max(predict_GD)+10, 'color', 'r');
title('Plot for Gradient Descent Method');
legend('GD Method', 'y=x');
xlabel('observed/true values');
ylabel('predicted values Using Gradient Descent Method');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======= Part 7: Prediction Using Normal Equation

%   find the theta using the normal method
theta_Normal= normalEqn([ ones(size(X, 1), 1) X], y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta_Normal);
fprintf('\n');

temp =predict_X;
% add intercept term
temp=[ones(size(temp,1), 1) temp];

% predicting the values for predict_X using the theta obtained
% using Normal Method.
predict_Normal=temp*theta_Normal;

% Train Accuracy
fprintf('Train Accuracy Using Normal Method: %f\n', mean(double(predict_Normal == predict_y)) * 100);


% compare the predict_GD and predict_y

diff_Normal=abs(predict_Normal-predict_y);

figure;

plot( predict_y,predict_Normal, 'color' , 'b');
hold on;
plot(0:max(predict_Normal)+10, 0:max(predict_Normal)+10, 'color', 'r');
title('Plot for Normal Method');
legend('Normal Method', 'y=x');
xlabel('observed/true values');
ylabel('predicted values using Normal Method');

fprintf('Program paused. Press enter to continue.\n');
pause;
close all;
%% ========= Part 8: Compare GD Method and Normal Method =====================
figure;
plot(predict_GD, predict_Normal, 'color', 'g');
title('Compare GD and Normal Method');
hold on;
plot(0:max(predict_GD)+10, 0:max(predict_GD)+10, 'color', 'r');
legend('Compare the two', 'y=x');
xlabel('predicted values using Gradient Descent Method');
ylabel('predicted values using Normal Method');

fprintf('Program paused. Press enter to continue.\n');
pause;
close;

figure;
subplot(3,1,1);
plot( 1:size(predict_y, 1), predict_y, 'color', 'b');
title('predict_y');
xlabel('Verifying Data');
ylabel('predict\_y');

subplot(3,1,2);
plot( 1:size(predict_y, 1), predict_GD, 'color', 'b');
title('predict_GD');
xlabel('Verifying Data');
ylabel('predict\_GD');

subplot(3,1,3);
plot( 1:size(predict_y, 1), predict_Normal, 'color', 'b');
title('predict_Normal');
xlabel('Verifying Data');
ylabel('predict\_Normal');

req= [predict_y predict_GD predict_Normal];
theta= [theta_GD theta_Normal];

fprintf('Program paused. Press enter to continue.\n');
pause;
close all;
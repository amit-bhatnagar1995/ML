function plotData(X, y, pausetime)
    for i=1:size(X, 2)
        %file_name=strcat('feature ', num2str(i));
        %file_name= [file_name '.jpg'];
        plot(X(:, i), y, 'rx', 'MarkerSize', 10);
        xlabel(['feature ', num2str(i)]);
        ylabel('no of shares ');
        %saveas(gcf, file_name);
        pause(pausetime);
    end
    
end
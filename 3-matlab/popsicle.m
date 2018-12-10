clear all; close all;

% Grid Size
grid_dim = int8(10);

% Number of Simulations to run
N = 1000000;

% Counter for number of sticks drawn
total = double.empty();

% Start iterating
tic
for jj=1:N
    % No pizza party to begin with
    pizza_party = 0;
    
    % Shuffle the popsicle sticks into the cup
    popsicle_sticks = randperm(grid_dim*grid_dim);
    
    % Erase the previous grid to start fresh
    number_grid = zeros(grid_dim,'int8');
    
    % pull out the first stick
    good_behavior_count = int8(1);
    
    % Keep drawing sticks until there is a pizza party
    while pizza_party == 0
        number_grid(popsicle_sticks(good_behavior_count)) = 1;
        
        % don't check for a complete row if we haven't drawn enough for
        % complete row.
        if good_behavior_count >= grid_dim
           col = idivide(popsicle_sticks(good_behavior_count)-1,grid_dim)+1;
           row = mod(popsicle_sticks(good_behavior_count),grid_dim)+1;
           count_row = 0;
           count_col = 0;
           for ii=(col-1)*grid_dim+1:(col-1)*grid_dim+grid_dim
               count_col = count_col + number_grid(ii);
           end
           for ii=0:9
              count_row = count_row + number_grid(ii*grid_dim+row);
           end
           if count_row == grid_dim || count_col == grid_dim
               pizza_party = 1;
               total(jj) = double(good_behavior_count);
           end
        end
        good_behavior_count = good_behavior_count + 1;
    end
end
stop_timer = toc;

histogram(total,'BinWidth',1)

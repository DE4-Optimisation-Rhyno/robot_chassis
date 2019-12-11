clc 
clear all 
%-----------Notes
%As four materials are being tested individually, many functions need to be
%repeated for variables relative to each material. To simplify this, when
%material data is loaded in it comes in the order (1:Aluminium 6061, 2:Zinc Alloy, 3: Magneusium Alloy, 4:Titanium Grade four 
%all of the warnings in the code refer to pre-allocating variables
%------------Latin Hypercube Sampling to define DOE------------------------
LH = lhsdesign(10,3, 'criterion', 'maximin'); %use lhsdesign for DOE
min_width = 253.2;
max_width = 660;
min_length = 300;
max_length = 1000;
min_thickness = 1.6;
max_thickness = 30;
width_range = max_width - min_width; 
length_range = max_length - min_length;
thickness_range = max_thickness - min_thickness;
%---sampling points for materials------------------------------------------
lhs_length = min_length + ((LH(:,1))*length_range); %LH has returned a random point between 1 and 0 which I then multiply by the range and add to the lowest bound 
lhs_width = min_width + ((LH(:,2))*width_range);
lhs_thickness = min_thickness + ((LH(:,3))*thickness_range);
LHS_L_W_T = [lhs_length, lhs_width, lhs_thickness];

%as this is random each time, used values can be seen in the csv file
%-----------------------Actual Work Now------------------------------------
%-----------------organising data by material------------------------------
Sim_data = csvread('Formatted_Data_Norms.csv',1,1,[1,1,40,12]);
Design_variables = (Sim_data(1:10,[1,2,3])); %design variables are the same for all materials so Design_variables can be used for all optimisation problems 

Al_outputs = (Sim_data(1:10,[4,6])); %seperate output data into its correspoinding mateirals 
Zn_outputs = (Sim_data(11:20,[4,6]));
Mg_outputs = (Sim_data(21:30,[4,6]));
Ti_outputs = (Sim_data(31:40,[4,6]));

Al_data = [Design_variables,Al_outputs]; % combine input and output data for materials for easier work
Zn_data = [Design_variables,Zn_outputs];
Mg_data = [Design_variables,Mg_outputs];
Ti_data = [Design_variables,Ti_outputs];

%this is now stored in a 3 dimensional array where each set of material's
%data is on a different page

Material_data(:,:,1) = (Al_data); %page 1 
Material_data(:,:,2) = (Zn_data); %page 2 
Material_data(:,:,3) = (Mg_data); %page 3 
Material_data(:,:,4) = (Ti_data); %page 4 

rng(1);  %with a random seed 1
%----randomise data and then split into with a ratio of 80:20 to be used as
%training and test data 
for i = 1:4
    Rand_order = randperm(length(Design_variables));
    Material_data_rand(:,:,i) = Material_data(Rand_order,:,i);
    Test_points = ceil(0.8*length(Material_data_rand));
    Train_data(:,:,i) = Material_data_rand(1:Test_points,:,i);
    Test_data(:,:,i) = Material_data_rand(Test_points+1:end,:,i);
end
%--------------------normalise data----------------------------------------
for i = 1:4
    [Design_inputs_training_norm(:,:,i),Input_PS(i)] = mapstd(Train_data(:,1:3,i)');  %normalise data with the mapstd function, it is important that the inverse of matrices is used as mapstd normalises rows not columsn  
    [Outputs_mass_training_norm(:,:,i),Output_mass_PS(i)] = mapstd(Train_data(:,4,i)');
    [Outputs_fos_training_norm(:,:,i),Output_fos_PS(i)] = mapstd(Train_data(:,5,i)');
    
    Design_inputs_testing_norm(:,:,i) = mapstd('apply',Test_data(:,1:3,i)',Input_PS(i)); %use corresponding processing settings between the training and test data 
    Outputs_mass_testing_norm(:,:,i) = mapstd('apply',Test_data(:,4,i)',Output_mass_PS(i));
    Outputs_fos_testing_norm(:,:,i) = mapstd('apply', Test_data(:,5,i)',Output_fos_PS(i));
    
end

% for i = i:4
%   New_Design_inputs_training_norm(:,:,i) = Design_inputs_training_norm(:,:,i)'; %for ease of work 
%   New_Outputs_mass_training_norm(:,:,i) = Outputs_mass_training_norm(:,:,i)';
%   New_Outputs_fos_training_norm(:,:,i) = Outputs_fos_training_norm(:,:,i)';
%   New_Design_inputs_testing_norm(:,:,i) = Design_inputs_testing_norm(:,:,i)';
%   New_Outputs_mass_testing_norm(:,:,i) = Outputs_mass_testing_norm(:,:,i)';
%   New_Outputs_fos_testing_norm(:,:,i) = Outputs_fos_testing_norm(:,:,i)';
% end 

%--------------Observing relationship between normalised data--------------
%plot normalised data points to prove that there is a linear relationship
%between variables and outputs
figure(1)
set(gcf,'color','w'); 
sgtitle('Scatter plots of normalised Aluminium Simulation Data')
subplot(1,2,1)
axis('square')
scatter(Design_inputs_training_norm(1,:,1),Outputs_mass_training_norm(:,:,1))
title("Length against Mass")
xlabel('length')
ylabel('mass')
pbaspect([1 1 1])
subplot(1,2,2) 
axis('square')
scatter(Design_inputs_training_norm(1,:,1),Outputs_fos_training_norm(:,:,1))
title("Length against FOS")
xlabel('length')
ylabel('Safety Factor')
pbaspect([1 1 1])

figure(2)
set(gcf,'color','w'); 
sgtitle('Scatter plots of normalised Aluminium Simulation Data')
subplot(1,2,1)
axis('square')
scatter(Design_inputs_training_norm(2,:,1),Outputs_mass_training_norm(:,:,1))
title("Width against Mass")
xlabel('Width')
ylabel('mass (kg)')
pbaspect([1 1 1])
subplot(1,2,2) 
axis('square')
scatter(Design_inputs_training_norm(2,:,1),Outputs_fos_training_norm(:,:,1))
title("Width against FOS")
xlabel('Width')
ylabel('Safety Factor')
pbaspect([1 1 1])

figure(3)
set(gcf,'color','w'); 
sgtitle('Scatter plots of normalised Aluminium Simulation Data')
subplot(1,2,1)
axis('square')
scatter(Design_inputs_training_norm(3,:,1),Outputs_mass_training_norm(:,:,1))
title("Normalised Thickness against Mass")
xlabel('Normalised Thickness')
ylabel('Normalised Mass')
pbaspect([1 1 1])
subplot(1,2,2) 
axis('square')
scatter(Design_inputs_training_norm(3,:,1),Outputs_fos_training_norm(:,:,1))
title("Normalised Thickness against FS")
xlabel('Normalised thickness')
ylabel('Normalised FS')
pbaspect([1 1 1])

%---------------------Linear Regression------------------------------------

for i = 1:4
    Betas_mass(:,:,i) = mvregress(Design_inputs_training_norm(:,:,i)',Outputs_mass_training_norm(:,:,i)'); %calculating Beta values with mvregress
    Betas_fos(:,:,i) = mvregress(Design_inputs_training_norm(:,:,i)',Outputs_fos_training_norm(:,:,i)');
    
    LM_mass{i} = fitlm(Design_inputs_training_norm(:,:,i)',Outputs_mass_training_norm(:,:,i)');%added in to check that mvregress and calculated beta values are correct 
    LM_fos{i} = fitlm(Design_inputs_training_norm(:,:,i)',Outputs_fos_training_norm(:,:,i)'); 
end

%------------Additional R squared Calulations for each material-----------

for i = 1:4
    R_squared_mass(i) = 1 - norm(Design_inputs_testing_norm(:,:,i)'*Betas_mass(:,:,i) - Outputs_mass_testing_norm(:,:,i)')^2/norm(Outputs_mass_testing_norm(:,:,i)' - mean(Design_inputs_testing_norm(:,:,i)'))^2;
    R_squared_fos(i) = 1 - norm(Design_inputs_testing_norm(:,:,i)'*Betas_fos(:,:,i) - Outputs_fos_testing_norm(:,:,i)')^2/norm(Outputs_fos_testing_norm(:,:,i)' - mean(Design_inputs_testing_norm(:,:,i)'))^2;
end
%allign with results from fitlm and are all acceptable values 

%------------define all functions 
    
Fun_mass{1} = @(x)  Betas_mass(1,:,1)*x(1) + Betas_mass(2,:,1)*x(2) + Betas_mass(3,:,1)*x(3); %functions were required to be defined seperately  
Fun_fos{1} = @(x) -1*(Betas_fos(1,:,1)*x(1) + Betas_fos(2,:,1)*x(2) + Betas_fos(3,:,1)*x(3));
    
Fun_mass{2} = @(x)  Betas_mass(1,:,2)*x(1) + Betas_mass(2,:,2)*x(2) + Betas_mass(3,:,2)*x(3); 
Fun_fos{2} = @(x) -1*(Betas_fos(1,:,2)*x(1) + Betas_fos(2,:,2)*x(2) + Betas_fos(3,:,2)*x(3));
    
Fun_mass{3} = @(x)  Betas_mass(1,:,3)*x(1) + Betas_mass(2,:,3)*x(2) + Betas_mass(3,:,3)*x(3); 
Fun_fos{3} = @(x) -1*(Betas_fos(1,:,3)*x(1) + Betas_fos(2,:,3)*x(2) + Betas_fos(3,:,3)*x(3));
    
Fun_mass{4} = @(x)  Betas_mass(1,:,4)*x(1) + Betas_mass(2,:,4)*x(2) + Betas_mass(3,:,4)*x(3); 
Fun_fos{4} = @(x) -1*(Betas_fos(1,:,4)*x(1) + Betas_fos(2,:,4)*x(2) + Betas_fos(3,:,4)*x(3));

%these are the functions used for multi-objective optimisation when using
%the gamultiobj function 
Fun{1} = @(x) [Betas_mass(1,:,1)*x(1) + Betas_mass(2,:,1)*x(2) + Betas_mass(3,:,1)*x(3),-1*(Betas_fos(1,:,1)*x(1) + Betas_fos(2,:,1)*x(2) + Betas_fos(3,:,1)*x(3))];
Fun{2} = @(x) [Betas_mass(1,:,2)*x(1) + Betas_mass(2,:,2)*x(2) + Betas_mass(3,:,2)*x(3),-1*(Betas_fos(1,:,2)*x(1) + Betas_fos(2,:,2)*x(2) + Betas_fos(3,:,2)*x(3))];
Fun{3} = @(x) [Betas_mass(1,:,3)*x(1) + Betas_mass(2,:,3)*x(2) + Betas_mass(3,:,3)*x(3),-1*(Betas_fos(1,:,3)*x(1) + Betas_fos(2,:,3)*x(2) + Betas_fos(3,:,3)*x(3))];
Fun{4} = @(x) [Betas_mass(1,:,4)*x(1) + Betas_mass(2,:,4)*x(2) + Betas_mass(3,:,4)*x(3),-1*(Betas_fos(1,:,4)*x(1) + Betas_fos(2,:,4)*x(2) + Betas_fos(3,:,4)*x(3))];

%------Setting up for optimisation functions 

options1 = optimoptions('fmincon','Algorithm','interior-point'); %interior points is the default solver and recommended for first passes
options2 = optimoptions('fmincon','Algorithm','sqp'); %SQP is suited for speed on smaller optmisation problems (like this one)

%following values are common to all minimisation problems 
%Inequality constraints
A = [0, -1, 2];
b = [-1.4]; %normalised inequality constraint

%Nonlinear constraints (none)
Aeq = [];
beq = [];
%specify no nonlinear constraints 
nonlcon = [];

%individually minimise mass and fos functions with multiple methods (interior points, SQP and GA)

for i = 1:4
    %initial guesses are the maximum values used in the samples data 
    IG = [max(Design_inputs_training_norm(:,:,i)')];
    
    %Upper and lower bounds of variables 
    Lb = mapstd('apply',[300 253.2 1.6]',Input_PS(i))'; %applying normalisation to upper and lower bound values 
    Ub = mapstd('apply',[1000 660 30]',Input_PS(i))';
  
    [mass_dims_int{i}, mass_int{i}] = fmincon(Fun_mass{i},IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options1);
    [mass_dims_sqp{i}, mass_sqp{i}] = fmincon(Fun_mass{i},IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options2);
    [mass_dims_ga{i}, mass_ga{i}] = ga(Fun_mass{i},3,A,b,Aeq,beq,Lb,Ub,nonlcon);
    
    [fos_dims_int{i}, fos_int{i}] = fmincon(Fun_fos{i},IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options1);
    [fos_dims_sqp{i}, fos_sqp{i}] = fmincon(Fun_fos{i},IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options2);
    [fos_dims_ga{i}, fos_ga{i}] = ga(Fun_fos{i},3,A,b,Aeq,beq,Lb,Ub,nonlcon); 
end 
    
%For timing individual functions 
% tic
% for i = 1:4
%     %initial guesses are the maximum values used in the samples data 
%     IG = [max(Design_inputs_training_norm(:,:,i)')];
%     
%     %Upper and lower bounds of variables 
%     Lb = mapstd('apply',[300 253.2 1.6]',Input_PS(i))'; %applying normalisation to upper and lower bound values 
%     Ub = mapstd('apply',[1000 660 30]',Input_PS(i))';
%     
%     [mass_dims_int{i}, mass_int{i}] = fmincon(Fun_mass{i},IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options1); % comment out functions to test individual ones for time 
%     [mass_dims_int{i}, mass_int{i}] = fmincon(Fun_mass{i},IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options2);
%     [mass_dims_ga{i}, mass_ga{i}] = ga(Fun_mass{i},3,A,b,Aeq,beq,Lb,Ub,nonlcon);
%      
% end 
% toc
%----remove normalisation from results to get true optimum values---------

for i = 1:4
    %----------optimum mass data 
    Optimum_mass_dims_int{i} = mapstd('reverse',mass_dims_int{i}',Input_PS(i)); %make sure the corresponding PS are used for proper de-normalisation 
    Optimum_mass_dims_sqp{i} = mapstd('reverse',mass_dims_sqp{i}',Input_PS(i));
    Optimum_mass_dims_ga{i} = mapstd('reverse',mass_dims_ga{i}',Input_PS(i));
    
    Optimum_mass_int{i} = mapstd('reverse',mass_int{i}',Output_mass_PS(i));
    Optimum_mass_sqp{i} = mapstd('reverse',mass_sqp{i}',Output_mass_PS(i));
    Optimum_mass_ga{i} = mapstd('reverse',mass_ga{i}',Output_mass_PS(i));
    
    %----------optimum fos data 
    Optimum_fos_dims_int{i} = mapstd('reverse',fos_dims_int{i}',Input_PS(i));
    Optimum_fos_dims_sqp{i} = mapstd('reverse',fos_dims_sqp{i}',Input_PS(i));
    Optimum_fos_dims_ga{i} = mapstd('reverse',fos_dims_ga{i}',Input_PS(i));
    
    Optimum_fos_int{i} = mapstd('reverse',fos_int{i}',Output_fos_PS(i));
    Optimum_fos_sqp{i} = mapstd('reverse',fos_sqp{i}',Output_fos_PS(i));
    Optimum_fos_ga{i} = mapstd('reverse',fos_ga{i}',Output_fos_PS(i));
    
end 
%-----Multi_objective optimisation 

for i = 1:4 
    Lb = mapstd('apply',[300 253.2 1.6]',Input_PS(i))'; %applying normalisation to upper and lower bound values 
    Ub = mapstd('apply',[1000 660 30]',Input_PS(i))';
    [parameter_vals{i},function_vals{i}] = gamultiobj(Fun{i},3,A,b,[],[],Lb,Ub,nonlcon);   %gamultiobj returns a points on the pareto set  
end

%-----show pareto set of lowest mass design material 
figure(4)
set(gcf,'color','w'); 
Opti_masses = mapstd('reverse',function_vals{3}(:,1)',Output_mass_PS(3))';
Opti_SF = mapstd('reverse',function_vals{3}(:,2)',Output_fos_PS(3))';
plot(function_vals{3}(:,1),function_vals{3}(:,2),'ko')
%plot(Opti_masses,Opti_SF,'ko') %plot of un-normalised data 
xlabel('mass')
ylabel('safety factor')
title('Pareto Points of Normalised Optimum Values for Magnesium Alloy')

%----Reverse normalisation for the found pareto points 
for i = 1:4
    Pareto_points{i} = [mapstd('reverse',function_vals{i}(:,1)',Output_mass_PS(i))', mapstd('reverse',function_vals{i}(:,2)',Output_fos_PS(i))',  mapstd('reverse',parameter_vals{i}',Input_PS(i))'];
end 

%----applying a weighted sum to find an optimal solution of Magnesium alloy------ 
%a function is made up of the two other objective functions are they are
%then weighted, the two weightings need to add to 1 
f = @(x) 0.5*(Betas_mass(1,:,3)*x(1) + Betas_mass(2,:,3)*x(2) + Betas_mass(3,:,3)*x(3)) - 0.5*(Betas_fos(1,:,3)*x(1) + Betas_fos(2,:,3)*x(2) + Betas_fos(3,:,3)*x(3));
IG = [max(Design_inputs_training_norm(:,:,3)')];
Lb = mapstd('apply',[300 253.2 1.6]',Input_PS(3))'; %applying normalisation to upper and lower bound values 
Ub = mapstd('apply',[1000 660 30]',Input_PS(3))';
[Opti_dims,Opti_vals] = fmincon(f,IG,A,b,Aeq,beq,Lb,Ub,nonlcon,options1);

MultiObj_opti = mapstd('reverse',Opti_dims',Input_PS(3));%from solidworks this gives a mass of 45 kg and a FOS of 57, seems sensible 

%-------Sensitivity analysis of one material (Aluminium)-------------------
IG = [max(Design_inputs_training_norm(:,:,1)')];
Lb_sens_1 = mapstd('apply',[301.5 254.5 1.61]',Input_PS(1))'; %paramters are varied by + 0.5%
Ub_sens_1 = mapstd('apply',[1005 663.3 30.15]',Input_PS(1))';

Lb_sens_2 = mapstd('apply',[303 255.7 1.62]',Input_PS(1))'; %parameters are varied by  + 1%
Ub_sens_2 = mapstd('apply',[1010 667 30.3]',Input_PS(1))';

[mass_dims_sens_1, mass_sens_1] = fmincon(Fun_mass{1},IG,A,b,Aeq,beq,Lb_sens_1,Ub_sens_1,nonlcon,options1);%sensitivity of mass
[mass_dims_sens_2, mass_sens_2] = fmincon(Fun_mass{1},IG,A,b,Aeq,beq,Lb_sens_2,Ub_sens_2,nonlcon,options1);

[fos_dims_sens_1, fos_sens_1] = fmincon(Fun_fos{1},IG,A,b,Aeq,beq,Lb_sens_1,Ub_sens_1,nonlcon,options1);%sensitivity of FS
[fos_dims_sens_2, fos_sens_2] = fmincon(Fun_fos{1},IG,A,b,Aeq,beq,Lb_sens_2,Ub_sens_2,nonlcon,options1);

Optimum_mass_dims_sens_1 = mapstd('reverse',mass_dims_sens_1',Input_PS(1));  %new optimised values are printed and checked outisde of matlab 
Optimum_mass_sens_1 = mapstd('reverse',mass_sens_1',Output_mass_PS(1)); 
Optimum_mass_dims_sens_2 = mapstd('reverse',mass_dims_sens_2',Input_PS(1));  
Optimum_mass_sens_2 = mapstd('reverse',mass_sens_2',Output_mass_PS(1)); 

Optimum_fos_dims_sens_1 = mapstd('reverse',fos_dims_sens_1',Input_PS(1));  
Optimum_fos_sens_1 = mapstd('reverse',fos_sens_1',Output_fos_PS(1)); 
Optimum_fos_dims_sens_2 = mapstd('reverse',fos_dims_sens_2',Input_PS(1));  
Optimum_fos_sens_2 = mapstd('reverse',fos_sens_2',Output_fos_PS(1)); 
%------Print Optimum values from all studies 
%optimum values and corresponding data are printed here
disp('optimum masss dimensions for Al 6061, Zn Ag40, Mg AZ31B and Ti R50700 are as follows in order')
disp(Optimum_mass_dims_sqp{1})
disp(Optimum_mass_dims_sqp{2})
disp(Optimum_mass_dims_sqp{3})
disp(Optimum_mass_dims_sqp{4})

disp('optimum factory of safety dimensions for Al 6061, Zn Ag40, Mg AZ31B and Ti R50700 are as follows in order')
disp(Optimum_fos_dims_sqp{1})
disp(Optimum_fos_dims_sqp{2})
disp(Optimum_fos_dims_sqp{3})
disp(Optimum_fos_dims_sqp{4})

disp('Sensitivity results for a 0.5% perturbation for Aluminium alloy show follwing optimisers for mass and FS respectively')
disp(Optimum_mass_dims_sens_1)
disp(Optimum_fos_dims_sens_1)

disp('Sensitivity results for a 1% perturbation for Aluminium alloy show follwing optimisers for mass and FS respectively')
disp(Optimum_mass_dims_sens_2)
disp(Optimum_fos_dims_sens_2)
%Print Multiobjective Optimisation results 

disp('Multiobjective optimisers of magnesium alloy are')
disp(MultiObj_opti)





%% CS229A Main Module
clear;
clc;
close all;

% Part 1: Load Examples
% 1. load table from csv file
P14 = load('p14.mat');
appdata = P14.applicationdata;
appdata = sortrows(appdata,'filing_date','ascend');

% 3. remove NANs
appdata = rmmissing(appdata);
m = height(appdata);

% remove data before 2000s
t_data_2000 = datetime('1-1-2000','InputFormat','dd-MM-yyyy');
pend_indices_b4_2000 = find(appdata.filing_date<t_data_2000);
%appdata(pend_indices_b4_2000,:) = [];

% 2. Remove pending examples that have been pending less than 32 months from dec 1.2014 %% 
t_data_2yrs = datetime('1-4-2012','InputFormat','dd-MM-yyyy');
pend_indices_after2012 = find(appdata.filing_date>t_data_2yrs); % & appdata.disposal_type == 'PEND');
remove_indices = [pend_indices_b4_2000 ; pend_indices_after2012];
% remove indices AFTER the feature set are generated.
%appdata(remove_indices,:) = [];

% 4. convert table to array
% 4a. remove unique patent ID and dates. Set will still be ordered
appdata_arraytable = appdata(1:m,3:9);
labels = (appdata_arraytable(1:m,7));
label_ex = table2array(labels);
labels_num = grp2idx(label_ex); 

% 4b. generate the result label labels_num is result label where...
%   0 is abandoned, 1 is issued, 2 is pending
labels_num = labels_num - ones(m,1);

% 4c. generate a matrix with the numerical data excluding y labels

appdata_mat = table2array(appdata_arraytable(1:m,1:6)); 
%appdata_mat = appdata_mat(1:(m-n),:);

% 4d. add the numerical y labels
appdata_mat = [appdata_mat labels_num];
% examiner_id, examiner_art_unit, uspc_class,uspc_subclass,customer_number,appl_status_code,disposal_type
 
%% Test
m = size(appdata_mat,1);
y = appdata_mat(:,7);

% 3f. remove examples from last few years, because of the pending label.
% n = 3e6; 
% appdata_mat = appdata_mat(1:(m-n),:);
figure
histogram(appdata_mat(:,7))
title('Plot of Abandoned vs. Issued for Data');
xlabel('0: Abandoned, 1: Issued, 2: pending > 2 years');
ylabel('Number of Patents, Data');
%https://www.uspto.gov/web/offices/ac/ido/oeip/taf/reports.htm#by_type

%% Part 2: Generate numerical features
% Use MATLAB Maps: 
% https://www.mathworks.com/help/matlab/ref/containers.map.html...
% ?searchHighlight=map%20object&s_tid=doc_srchtitle

%1. Create feature maps

%feature 1a map: exam_exp
M_exam_exp = containers.Map('KeyType','double','ValueType','double');
%feature 1b map: exam_allowed
M_exam_all = containers.Map('KeyType','double','ValueType','double');
%feature 2 map: exam_art_sat
M_exam_art_sat = containers.Map('KeyType','double','ValueType','double');
%feature 3 map: class_sat
M_class_sat = containers.Map('KeyType','double','ValueType','double');
%feature 4 map: subclass_sat
M_subclass_sat = containers.Map('KeyType','double','ValueType','double');
%feature 5a map: firm_exp
M_firm_exp =  containers.Map('KeyType','double','ValueType','double');
%feature 5b map: firm_iss
M_firm_iss =  containers.Map('KeyType','double','ValueType','double');
%feature 6: allowance rate: exam_all/exam_exp for examiner
%feature 7: firm issue rate: firm_iss/firm experience

%2. initialize X
X  = zeros(m,7);

for i = 1:size(appdata_mat,1)
    %feature 1 & 6: exam_exp, exam_allowance rate
    if isKey(M_exam_exp,appdata_mat(i,1))
        X(i,1) = M_exam_exp(appdata_mat(i,1));
        X(i,6) = M_exam_all(appdata_mat(i,1))/M_exam_exp(appdata_mat(i,1));
        M_exam_exp(appdata_mat(i,1)) = M_exam_exp(appdata_mat(i,1)) +1;
        if y(i) == 1
            M_exam_all(appdata_mat(i,1)) = M_exam_all(appdata_mat(i,1)) +1;
        end
    else
        M_exam_exp(appdata_mat(i,1)) = 1;
        if y(i) == 1
            M_exam_all(appdata_mat(i,1)) =  1;
        else
            M_exam_all(appdata_mat(i,1)) =  0;
        end
    end
    %feature 2: exam_art_sat
    if isKey(M_exam_art_sat,appdata_mat(i,2))
        X(i,2) = M_exam_art_sat(appdata_mat(i,2));
        M_exam_art_sat(appdata_mat(i,2)) = M_exam_art_sat(appdata_mat(i,2))+1; 
    else
       M_exam_art_sat(appdata_mat(i,2)) = 1;
    end
    %feature 3: class_sat
    if isKey(M_class_sat,appdata_mat(i,3))
        X(i,3) = M_class_sat(appdata_mat(i,3));
        M_class_sat(appdata_mat(i,3)) = M_class_sat(appdata_mat(i,3)) + 1;
    else
        M_class_sat(appdata_mat(i,3)) = 1;
    end
    %feature 4: subclass_sat
    if isKey(M_subclass_sat,appdata_mat(i,4))
        X(i,4) = M_subclass_sat(appdata_mat(i,4));
        M_subclass_sat(appdata_mat(i,4)) = M_subclass_sat(appdata_mat(i,4))+1;
    else
        M_subclass_sat(appdata_mat(i,4)) = 1;
    end
    %feature 5 & 7: firm_exp & firm success rate
    if isKey(M_firm_exp,appdata_mat(i,5))
        X(i,5) = M_firm_exp(appdata_mat(i,5));
        X(i,7) = M_firm_iss(appdata_mat(i,5))/M_firm_exp(appdata_mat(i,5));
        M_firm_exp(appdata_mat(i,5)) =  M_firm_exp(appdata_mat(i,5)) + 1;
        if y(i) == 1
            M_firm_iss(appdata_mat(i,5)) = M_firm_iss(appdata_mat(i,5)) + 1;
        end
    else
        M_firm_exp(appdata_mat(i,5)) = 1;
        if y(i) == 1 
            M_firm_iss(appdata_mat(i,5)) = 1;
        else
            M_firm_iss(appdata_mat(i,5)) = 0; %assign the firm to map with zero score
        end
    end
end

% Select for 2000-2012 data only to train on to prevent skewed data
X(remove_indices,:) = [];
%%
y(remove_indices,:) = [];
m_new = size(X,1);
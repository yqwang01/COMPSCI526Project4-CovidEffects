/*------------------------------------------------------------------------------
	This STATA do file runs t-test to check if average scores are different
	within students belonging to different groups. For instance, students
	infected with Covid-19 vs not-infected, Male students vs Female students, etc.
	
	This code is fully reproducable after changing directory to save
	results on line 25. 
	No need to change anything to load dataset (!requires internet!)

	Prints four text files with t-test results:
		1. ttest_covidpos (COVID-19 infected vs non-infected)
		2. ttest_freelunch (Freelunch receiver vs non-receiver)
		3. ttest_gender (Male vs Female)
		4. ttest_school (Poor Schools vs Rich Schools)
		
	Results from these tests are used on our the final report.

	Author: Anubhav Dhakal
------------------------------------------------------------------------------*/

clear all

// Directory to save regression tables (Change this to your need)
if "`c(username)'" == "ad641" global workspace "C:\Users\ad641\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science/problem_sets/Project Team 4/results"
if "`c(username)'" == "anubh" global workspace "C:\Users\anubh\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science/problem_sets/Project Team 4/results"

cd "${workspace}"

*-------------------------------------------------------------------------------	
**# Load the dataset-
import excel "https://github.com/yqwang01/COMPSCI526Project4-CovidEffects/raw/refs/heads/main/Datasets/COVID-19-Constructed-Dataset-(PANEL).xlsx", firstrow clear

*-------------------------------------------------------------------------------	
**# Cleaning dataset

* Relabeling
label define school 0 "Wealthy School" 1 "Poor School"
label values school school

replace gender = gender == 0
label define gender 1 "Female" 0 "Male"
label values gender gender

label define covidpos 1 "Yes" 0 "Null"
label values covidpos covidpos
la var covidpos "Covid (=1)"

gen learning_method = inlist(timeperiod, 3, 4, 5)
la var learning_method "Learning Method- in person or online"
label define learning_method 0 "In-person" 1 "Online"
lab values learning_method learning_method

la define edu_levels 1 "High School diploma" 2 "Bachelors" 3 "Masters" 4 "Doctoral"
la values fathereduc mothereduc edu_levels

gen hh_income = householdincome/1000, after(householdincome)
la var hh_income "HH income"

egen scoreSL = rowmean(readingscoreSL writingscoreSL mathscoreSL)

label define freelunch 1 "Freelunch = Yes" 0 "Freelunch = No"

*-------------------------------------------------------------------------------		
**# T-tests and writing results
foreach xvar in school gender covidpos freelunch{
	
	cap log close
	log using "ttest_`xvar'", replace text
		
	foreach yvar in readingscoreSL writingscoreSL mathscoreSL scoreSL {
		
		if "`yvar'" == "readingscoreSL" local v = "Reading Score"
		if "`yvar'" == "writingscoreSL" local v = "Writing Score"
		if "`yvar'" == "mathscoreSL" local v = "Math Score"
		if "`yvar'" == "scoreSL" local v = "Average of all Scores"
		
		di ""
		di ""
		di "`v'"
		ttest `yvar', by(`xvar')
	}

	cap log close
}


*-------------------------------------------------------------------------------	
**# End of Do file
exit
*-------------------------------------------------------------------------------
/*
This do file explores t-tests
Final version will be on Python!
*/

clear all

// Dataset directory
if "`c(username)'" == "ad641" global workspace "C:\Users\ad641\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science"
if "`c(username)'" == "anubh" global workspace "C:\Users\anubh\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science"

cd "${workspace}"

**# Load the dataset
import excel "problem_sets\Project Team 4\data\COVID-19-Constructed-Dataset-(PANEL).xlsx", sheet("Sheet1") firstrow clear


**# Cleaning dataset
* Relabeling
label define school 0 "Wealthy School" 1 "Poor School"
label values school school

label define gender 1 "Male" 0 "Female"
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

label define freelunch 1 "Yes" 0 "No"
label values freelunch freelunch


// Variable for one semester lag
gen time_1 = timeperiod - 1
preserve
	keep studentID *score* timeperiod
	rename timeperiod time_1
	rename (*score*) =_1
	
	tempfile tmp_time_lag
	save `tmp_time_lag'
restore
merge 1:1 studentID time_1 using `tmp_time_lag'
	assert time_1 == 5 if _merge == 2
	assert time_1 == -1 if _merge == 1
	drop if _merge==2
	drop time_1 _merge	
	
// Variable for two semester lag
gen time_2 = timeperiod - 2
preserve
	keep studentID *score* timeperiod
	rename timeperiod time_2
	rename (*score*) =_2
	
	tempfile tmp_time_lag
	save `tmp_time_lag'
restore
merge 1:1 studentID time_2 using `tmp_time_lag'
	assert inlist(time_2, 5, 4) if _merge == 2
	assert inlist(time_2, -1, -2) if _merge == 1
	drop if _merge==2
	drop time_2 _merge		
	
	
**# Log start
foreach xvar in school gender covidpos freelunch{
	
	cap log close
	log using "problem_sets\Project Team 4\log\ttest_`xvar'", replace	text
		
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


















































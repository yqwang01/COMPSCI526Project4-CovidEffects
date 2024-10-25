/*
This do file explores some regression
Final version will be written in Python!
*/

clear all

// Dataset directory
if "`c(username)'" == "ad641" global workspace "C:\Users\ad641\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science"
if "`c(username)'" == "anubh" global workspace "C:\Users\anubh\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science"

cd "${workspace}"

**# Log start
cap log close
log using "problem_sets\Project Team 4\log\regression_log_18oct24", replace

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

label define freelunch 1 "Freelunch = Yes" 0 "Freelunch = No"
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
	
**# Regression
foreach yvar in scoreSL { //readingscoreSL writingscoreSL mathscoreSL
	
	if "`yvar'" == "readingscoreSL" local v = "r"
	if "`yvar'" == "writingscoreSL" local v = "w"
	if "`yvar'" == "mathscoreSL" local v = "m"
	if "`yvar'" == "scoreSL" local v = "s"
	
	eststo clear
	
	qui{
		eststo `v'_reg_1: reg `yvar' covidpos i.gender i.school#c.hh_income c.hh_income familysize i.timeperiod, vce(cluster school)
		estadd local time_fe "\checkmark"		

	
		eststo `v'_reg_2: reg `yvar' covidpos i.gender i.school#c.hh_income c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		
		
		eststo `v'_reg_3: reg `yvar' covidpos i.gender i.school#c.hh_income c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc `yvar'_1, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		
		
		eststo `v'_reg_4: reg `yvar' covidpos i.gender i.school#c.hh_income c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc `yvar'_1 `yvar'_2, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"		
		
	}
		
	esttab , stat(r2 N time_fe father_edu mother_edu) se ///
				drop(*.timeperiod 0.school* 0.gender *educ) 
}


**# Writing table:
esttab  s_reg_1 s_reg_2 s_reg_3 s_reg_4 ///
		using "problem_sets/Project Team 4/tables/tab_scoreSL_reg.tex", replace noobs nomtitles ///
		title("Average State Level score and COVID-19 \label{tab:scoreSL_reg}") ///
		booktabs cells(b(fmt(%9.3f) star) se(par fmt(%9.3f))) ///
		starlevels(* .1 ** .05 *** .01) ///
		drop(*.timeperiod 0.school* 0.gender *educ) ///
		order(covidpos 1.gender hh_income 1.school#c.hh_income  ///
				familysize scoreSL_1 scoreSL_2 _cons) ///
		stats(r2 N time_fe father_edu mother_edu, label("R2" "Observations" "Time Fixed Effect" "Father's Education FE" "Mother's Education FE",) fmt(%9.3gc))  noeqlines ///
		collabels(none) label substitute(\_ _) ///
		varlabels(	covidpos "COVID (=1)" ///
					1.gender "Male (=1)"  ///
					1.school#c.hh_income "Poor School X Income" ///
					hh_income "Household Income"  ///
					familysize  "Family Size" ///
					scoreSL_1 "Score (t-1)" ///
					scoreSL_2 "Score (t-2)" ///
					_cons "Constant")  ///
		prehead("\begin{table}[htbp]\centering"	///
			"\addtocounter{table}{0}" ///
			"\scalebox{1}{"	///
			"\begin{threeparttable}[b]" ///
			"\caption{@title}"	///
			"\begin{tabular}{l*{@span}{c}}"	///
			"\toprule"			///
			"\addlinespace") ///
		postfoot("\bottomrule"	///
				"\end{tabular}"	///
				"The table reports the effect of Covid on average scores calculated using reading, writing and math scores.\\" ///
				"Standard errors are clustered at school level.\\" ///
				"\emph{Levels of significance}: *\$p<0.1\$, **\$p<0.05\$, ***\$p<0.01\$" ///
				"%\begin{tablenotes}"	///
				"" ///
				"%\end{tablenotes}"	///				
				"\end{threeparttable}"	///
				"}"				///	
				"\end{table}"	 )

// exit
				
**# FE regressions
foreach yvar in scoreSL { // readingscoreSL writingscoreSL mathscoreSL
	
	eststo clear
	
	qui{
		
		eststo reg1: reg `yvar' i.studentID, r
		estadd local time_fe ""
		estadd local individual_fe "\checkmark"	
		
		eststo reg2: reg `yvar' i.learning_method i.studentID, r
		estadd local time_fe ""			
		estadd local individual_fe "\checkmark"	
		
		eststo reg3: reg `yvar' i.learning_method covidpos i.gender c.hh_income i.school familysize i.fathereduc i.mothereduc i.studentID, r
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		estadd local individual_fe "\checkmark"	
		
		predict score_predict, xb
		
// 		eststo: reg `yvar' i.learning_method `yvar'_1 i.studentID, r
// 		estadd local time_fe ""		
// 		estadd local individual_fe "\checkmark"	
		
// 		eststo: reg `yvar' i.learning_method covidpos i.gender i.school#c.hh_income c.hh_income familysize i.studentID, r
// 		estadd local time_fe ""		
// 		estadd local individual_fe "\checkmark"	
//		
// 		eststo: reg `yvar' covidpos i.gender i.school#c.hh_income c.hh_income familysize i.studentID i.timeperiod, r
// 		estadd local time_fe "\checkmark"		
// 		estadd local individual_fe "\checkmark"	
		
	}
	
	esttab  reg1 reg2 ///
	using "problem_sets/Project Team 4/tables/tab_scoreSL_fereg.tex", replace noobs nomtitles ///
	title("Average State Level score and COVID-19 \label{tab:scoreSL_reg}") ///
	booktabs cells(b(fmt(%9.3f) star) se(par fmt(%9.3f))) ///
	starlevels(* .1 ** .05 *** .01) ///
	drop( 0.learning_method *studentID) ///
	/// order(1.learning_method covidpos 1.gender hh_income 1.school  ///
	///		familysize _cons) ///
	stats(r2 N time_fe father_edu mother_edu, label("R2" "Observations" "Time Fixed Effect" "Father's Education FE" "Mother's Education FE",) fmt(%9.3gc))  noeqlines ///
	collabels(none) label substitute(\_ _) ///
	varlabels(	0.learning_method "Online (=1)" ///
				covidpos "COVID (=1)" ///
				1.gender "Male (=1)"  ///
				1.school#c.hh_income "Poor School X Income" ///
				hh_income "Household Income"  ///
				familysize  "Family Size" ///
				scoreSL_1 "Score (t-1)" ///
				scoreSL_2 "Score (t-2)" ///
				_cons "Constant")  ///
	prehead("\begin{table}[htbp]\centering"	///
		"\addtocounter{table}{0}" ///
		"\scalebox{1}{"	///
		"\begin{threeparttable}[b]" ///
		"\caption{@title}"	///
		"\begin{tabular}{l*{@span}{c}}"	///
		"\toprule"			///
		"\addlinespace") ///
	postfoot("\bottomrule"	///
			"\end{tabular}"	///
			"The table reports the effect of Covid on average scores calculated using reading, writing and math scores.\\" ///
			"Standard errors are clustered at school level.\\" ///
			"\emph{Levels of significance}: *\$p<0.1\$, **\$p<0.05\$, ***\$p<0.01\$" ///
			"%\begin{tablenotes}"	///
			"" ///
			"%\end{tablenotes}"	///				
			"\end{threeparttable}"	///
			"}"				///	
			"\end{table}"	 )
			
	esttab , stat(r2 N time_fe individual_fe) se ///
				drop( *studentID 0.learning_method) label
}


** Bar figures:
graph box score_predict, by(learning_method) ytitle("Average Score") box(1, fcolor("34 139 141") lc(black) lw(thin))

exit






















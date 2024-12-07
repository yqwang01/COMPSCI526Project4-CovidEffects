/*------------------------------------------------------------------------------
	This STATA do file runs Interaction and Fixed effect regression.
	I did not use Python for this because the existing packages do 
	not produce regression tables of publishable quality.

	This code is fully reproducable after changing directory to save
	results on line 25. 
	No need to change anything to load dataset (!requires internet!)

	Prints two latex tables and a box plot figure:
		1. tab_scoreSL_int.tex (Interaction Effect models)
		2. tab_scoreSL_fereg.tex (Fixed Effect models)
		3. fig_learning_method_box.png (Box plot for Fixed Effect model prediction)
		
	These latex tables and figure are called on final report.

	Author: Anubhav Dhakal
------------------------------------------------------------------------------*/

clear all

// Directory to save regression tables (Change this to your need)
if "`c(username)'" == "ad641" global workspace "C:\Users\ad641\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science/problem_sets/Project Team 4/tables/"
if "`c(username)'" == "anubh" global workspace "C:\Users\anubh\OneDrive - Duke University\c_backups\adR6\Documents\Duke\duke courses\First Semester\Compsci 526, Data Science/problem_sets/Project Team 4/tables"

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
label values freelunch freelunch


* Variable for one semester lag
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
	
* Variable for two semester lag
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

*-------------------------------------------------------------------------------		
**# Interaction Effect Regression Models

eststo clear

foreach yvar in readingscoreSL writingscoreSL mathscoreSL scoreSL {
	
	if "`yvar'" == "readingscoreSL" local v = "r"
	if "`yvar'" == "writingscoreSL" local v = "w"
	if "`yvar'" == "mathscoreSL" local v = "m"
	if "`yvar'" == "scoreSL" local v = "s"
	
	qui{
		eststo `v'_reg_1: reg `yvar' covidpos gender c.school#c.hh_income c.hh_income familysize i.timeperiod, vce(cluster school)
		estadd local time_fe "\checkmark"		

	
		eststo `v'_reg_2: reg `yvar' covidpos gender c.school#c.hh_income c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		
		eststo `v'_reg_3: reg `yvar' covidpos gender school c.gender#c.school c.school#c.hh_income c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		
		eststo `v'_reg_4: reg `yvar' covidpos gender school c.gender#c.school c.school#c.hh_income c.gender#c.covidpos c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		
		eststo `v'_reg_5: reg `yvar' covidpos gender school c.gender#c.school c.school#c.hh_income c.gender#c.covidpos c.school#c.covidpos c.hh_income familysize i.timeperiod i.fathereduc i.mothereduc, vce(cluster school)
		estadd local time_fe "\checkmark"			
		estadd local father_edu "\checkmark"	
		estadd local mother_edu "\checkmark"	
		
		eststo `v'_reg_6: reg `yvar' covidpos gender school c.gender#c.school c.school#c.hh_income c.gender#c.covidpos c.school#c.covidpos c.hh_income i.timeperiod i.fathereduc i.mothereduc, vce(cluster school)
		estadd local time_fe "Yes"			
		estadd local father_edu "Yes"	
		estadd local mother_edu "Yes"	
		
	}
					
}

*-------------------------------------------------------------------------------	
**# Writing Interaction Effect results on table

	esttab s_reg_6 r_reg_6 w_reg_6 m_reg_6   ///
				using "tab_scoreSL_int.tex", replace noobs collabels(none) ///
				title("Sore and COVID-19 with interactions \label{tab:scoreSL_reg_int}") ///
				booktabs cells(b(fmt(%9.3f) star) se(par fmt(%9.3f))) ///
				starlevels(* .05 ** .01 *** .001) stats(r2 N time_fe father_edu mother_edu, label("R2" "Observations" "Time Fixed Effect" "Father's Education FE" "Mother's Education FE",) fmt(%9.3gc)) se ///
				drop(*.timeperiod  *educ) ///
				varlabels(	covidpos "COVID (=1)" ///
					gender "Female (=1)"  ///
					school "Poor School" ///
					c.gender#c.school "Female X Poor School" ///
					c.gender#c.covidpos "Female X Covid" /// 
					c.school#c.hh_income "Poor School X HHIncome" ///
					c.school#c.covidpos "Poor School X Covid" ///
					hh_income "HHIncome"  ///
					familysize  "Family Size" ///
					scoreSL_1 "Score (t-1)" ///
					scoreSL_2 "Score (t-2)" ///
					c.gender#c.hh_income "Female X HHIncome" ///
					_cons "Constant")  ///
				order(covidpos gender school hh_income c.gender#c.covidpos c.school#c.covidpos c.gender#c.school c.school#c.hh_income) ///
				prehead("\begin{table}[htbp]\centering"	///
					"\scalebox{1}{"	///
					"\begin{threeparttable}[b]" ///
					"\caption{@title}"	///
					"\begin{tabular}{l*{@span}{c}}"	///
					"\toprule"			///
					"\addlinespace") ///
				postfoot("\bottomrule"	///
						"\end{tabular}"	///
						"The table presents four regression models with average, reading, writing, and math test scores as dependent variables. The results are displayed as columns, with each row corresponding to a variable used in the regression model. For example, the \textit{Female} X \textit{Covid} row represents the interaction term between the two dummy variables, \textit{Female} and \textit{Covid}. The values in the table show the coefficients, while the values in parentheses indicate the standard errors of the coefficients. Stars denote whether the coefficients are statistically significantly different from zero.\\" ///
						"\emph{Levels of significance}: *$p<0.1$, **$p<0.05$, ***$p<0.01$" ///
						"\end{threeparttable}"	///
						"}"				///	
						"\end{table}"	 )

*-------------------------------------------------------------------------------	
**# Fixed Effect Regression Models

foreach yvar in scoreSL { // readingscoreSL writingscoreSL mathscoreSL
	
	eststo clear
	
	qui{
		
		eststo reg1: reg `yvar' i.studentID, r
		estadd local time_fe ""
		estadd local individual_fe "\checkmark"	
		
		eststo reg2: reg `yvar' i.learning_method i.studentID, r
		estadd local time_fe ""			
		estadd local individual_fe "\checkmark"	
		
		predict score_predict, xb
		
	}
	
*-------------------------------------------------------------------------------	
**# Writing Fixed Effect results on table

	esttab  reg1 reg2 ///
	using "tab_scoreSL_fereg.tex", replace noobs nomtitles ///
	title("Average State Level score and COVID-19 \label{tab:scoreSL_reg}") ///
	booktabs cells(b(fmt(%9.3f) star) se(par fmt(%9.3f))) ///
	starlevels(* .1 ** .05 *** .01) ///
	drop( 0.learning_method *studentID) ///
	stats(individual_fe r2 N, label("Individual FE" "R2" "Observations") fmt(%9.3gc))  noeqlines ///
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
		"\scalebox{1}{"	///
		"\begin{threeparttable}[b]" ///
		"\caption{@title}"	///
		"\begin{tabular}{l*{@span}{c}}"	///
		"\toprule"			///
		"\addlinespace") ///
	postfoot("\bottomrule"	///
			"\end{tabular}"	///
			"The table reports the effect of Covid on average scores calculated using reading, writing and math scores.\\" ///
			"Robust standard errors used.\\" ///
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

*-------------------------------------------------------------------------------
**# Box Plot for Fized Effect model
graph box score_predict, by(learning_method) ytitle("Average Predicted Score") box(1, fcolor("34 139 141") lc(black) lw(thin)) name(fig_learning_method_box, replace)

graph export "fig_learning_method_box.png" , name(fig_learning_method_box) replace

*-------------------------------------------------------------------------------
**# End of Do file
exit
*-------------------------------------------------------------------------------
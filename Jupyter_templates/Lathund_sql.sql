-------------------------------------------------------------------------------
-- TOOLS ----------------------------------------------------------------------
/*
Text editor:			Sublime text
Formating check:	https://poorsql.com/
Web services:			Google cloud
*/



-------------------------------------------------------------------------------
-- GENERAL APPROACH -----------------------------------------------------------
/*
Before you code
	1 Undertsand you data - critically important!
	2 Think through the problem
	3 Slice big problems into smaller parts
	4 Create conceptual solution and implementation plan

When you code
	1 Write readable code
	2 Start with inner subqueries 
	3 Once all subqueries work - merge

Before you execute
	1 Don't be lazy - check your code one more time

After you execute
	1 Confirm your code works as intended
	2 Always read the error message - they exist for a reason
*/



-------------------------------------------------------------------------------
-- UDNERSTANDING TABLE RELATIONSHIPS ------------------------------------------
/*
Primer on on entity-relationship diagram symbols: ERD_cheat_sheet.pdf

Always seek to:
	1 Understand the entity-relationship betwwen tables
	2 Know which tables have unique keys
	3 Know all colum value definitions, data types, value ranges and error
	values. Datetime time zone. 
*/



-------------------------------------------------------------------------------
-- BIGQUERY SQL SYNTAX AND EXECUTION ORDER ------------------------------------
/*
Syntax order:			Execution order:

SELECT					FROM
FROM 						WHERE
WHERE						GROUP BY
GROUP BY 				HAVING
HAVING 					SELECT
WINDOW					ORDER BY
ORDER BY 				LIMIT
LIMIT 					
*/



-------------------------------------------------------------------------------
-- DATA TYPES (BIGQUERY) ------------------------------------------------------
/*
https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types

Bool
String 
Integer (INT64 with alias INT, SMALLINT, INTEGER, BIGINT, TINYINT, BYTEINT)
Numeric (DECIMAL is an alias for NUMERIC. BIGDECIMAL is an alias for BIGNUMERIC.)
	Much higher precision than float. This type can represent decimal fractions exactly, and is suitable for financial calculations.
Floating point (FLOAT64)
	Binary floating point numbers may deviate slightly from the the actual 
	number they intend to represent.
Date
	No timezone
Time
	No timezone
Datetime
	No timezone
Timestamp
	Absolute point in time
Array
Struct
Interval
JSON
Bytes
*/



-------------------------------------------------------------------------------
-- ORDER BY STATEMENT ---------------------------------------------------------

-- Syntax: ORDER BY column1, column2, ... ASC|DESC
SELECT *
FROM `nod-sql-copy.emp.employees`
ORDER BY first_name, last_name ASC;

SELECT first_name
FROM employees.employees
ORDER BY first_name DESC, last_name DESC;



-------------------------------------------------------------------------------
-- DATE FUNCTIONS (BIGQUERY) --------------------------------------------------
/*
https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions

CURRENT_DATE([time_zone])
	Returns current date DATE
	time_zone is optional

EXTRACT(part FROM date_expression)
	Returns number of type INT64
	part = dayofweek, day, dayofyear, week (begins on sunday), isoweek (begins 
	on monday), month, quarter, year, isoyear

DATE(year, month, day) 
DATE(timestamp_expression[, time_zone])
DATE(datetime_expression)
	Purpose: Construct DATE from INT64 values
	Returns DATE

DATE_ADD(date_expression, INTERVAL int64_expression date_part)
	Adds a specified time interval to a DATE
	date_part = day, week, month, quarter, year

DATE_SUB(date_expression, INTERVAL int64_expression date_part)
	Subtracts a specified time interval from a DATE.
	date_part = day, week, month, quarter, year

DATE_DIFF(date_expression_a, date_expression_b, date_part)
	date_diff = day, week (week begins on sunday), isoweek (week begins on
	monday), month, quarter, year, isoyear
	Returns INT64

DATE_TRUNC(date_expression, date_part)
	Truncates the date to the specified granularity
	date_part = day, week (week begins on sunday), isoweek (week begins on
	monday), month, quarter, year, isoyear
*/

SELECT 
	CURRENT_DATE() AS the_date,
	EXTRACT(year FROM hire_date) AS year_hire,
	DATE(2016, 12, 25) AS date_ymd,
	DATE(DATETIME "2016-12-25 23:59:59") AS date_dt,
	DATE_ADD(DATE "2008-12-25", INTERVAL 5 DAY) AS five_days_later,
  DATE_DIFF(hire_date, birth_date, Year) AS hiring_age, 
  DATE_DIFF(hire_date, birth_date, month) AS hiring_age_months, 
	DATE_DIFF('2017-12-30', '2014-12-30', YEAR) AS year_diff
FROM `nod-sql-copy.emp.employees`;


-- Year, month when they were hired. Leaving YYYY-MM-01
SELECT
    DATE_TRUNC(hire_date, month) AS year_month_hired
FROM 
    emp.employees
ORDER BY
    year_month_hired DESC;



-------------------------------------------------------------------------------
-- CONDITIONAL STATEMENTS FOR WHERE AND HAVING (BIGQUERY) ---------------------
/*
https://cloud.google.com/bigquery/docs/reference/standard-sql/operators

=				Equal
!=, <>	Not equal
<				Less than
>				Greater than
<=			Less than or equal to
>=			Greater than or equal to

[NOT] LIKE		Value does [not] match the pattern specified
				A percent sign "%" matches any number of characters or bytes
				An underscore "_" matches a single character or byte
				You can escape "\", "_", or "%" using two backslashes. For 
				example, "\\%". If you are using raw strings, only a single 
				backslash is required. For example, r"\%".
[NOT] BETWEEN	Value is [not] within the range specified
[NOT] IN 		Value is [not] in the set of values specified
IS [NOT] NULL 	Value is [not] NULL
IS [NOT] TRUE 	Value is [not] TRUE
IS [NOT] FALSE 	Value is [not] FALSE

NOT, AND, OR
*/

SELECT *
FROM emp.employees
WHERE (hire_date BETWEEN "1999-01-01" AND "1999-12-31"
		AND first_name IN ("Maria","Elvis"))
	OR (first_name <> "Elvis"
		AND gender = "M"
		AND birth_date > "1964-01-01"
		AND emp_no < 100000);

-- Find names that begin with Mar followed by any number of characters or bytes
SELECT *
FROM emp.employees
WHERE first_name LIKE("Mar%");

-- Find names that contains Mar preceded and followed by any number of 
-- haracters or bytes
SELECT *
FROM emp.employees
WHERE first_name LIKE("%mar%");

-- Find snames that begin with Mar followed by any two letters
SELECT *
FROM emp.employees
WHERE first_name LIKE("Mar__");



-------------------------------------------------------------------------------
-- AGGREGATE FUNCTIONS (BIGQUERY) ---------------------------------------------
/*
An aggregate function is a function that summarizes the rows of a group into 
a single value

https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions

AVG(col)				Returns the average of non-NULL input values
COUNT(col)			Returns the number of rows in the input
COUNTIF(col)		Returns the count of TRUE values for expression
MAX(col)				Returns the maximum value of non-NULL expressions
MIN(col)				Returns the minimum value of non-NULL expressions
SUM(col)				Returns the sum of non-null values
ANY_VALUE(col)	Returns expression for some row chosen from the group. Which row is chosen is nondeterministic, not random.
*/

SELECT 
    ROUND(MAX(salary),2) AS max_salary,
    MIN(salary) AS min_salary,
    AVG(salary) AS mean_salary
FROM emp.salaries;

SELECT
    dept_no,
    ANY_VALUE(emp_no) AS rand_employee,
    COUNT(emp_no) AS no_of_emp
FROM emp.dept_emp
GROUP BY dept_no;



-------------------------------------------------------------------------------
-- SELECT STATEMENT (BIGQUERY) ------------------------------------------------
/*
SELECT *					Returns all rows
SELECT DISTINCT(col) 		A SELECT DISTINCT statement discards duplicate rows 
							and returns only the remaining rows
SELECT * EXCEPT (col)		A SELECT * EXCEPT statement specifies the names of 
							one or more columns to exclude from the result
SELECT * REPLACE (expr) 	Replace column name or quantity with new expression
*/

SELECT *,
	first_name AS mid_name
FROM `nod-sql-copy.emp.departments`;

SELECT DISTINCT(gender)
FROM emp.employees;

SELECT * EXCEPT (order_id)
FROM orders;

SELECT * REPLACE ("widget" AS item_name)
FROM orders;

SELECT * REPLACE (quantity/2 AS quantity)
FROM orders;



-------------------------------------------------------------------------------
-- GROUP BY -------------------------------------------------------------------
-- Always used together with Aggregate functions
-- HAVING requires that a GROUP BY clause is present
-- HAVING is like WHERE but for values calculated from using group by

SELECT
    dept_no,
    COUNT(emp_no) AS no_of_emp
FROM emp.dept_emp
GROUP BY dept_no
HAVING no_of_emp > 20000
ORDER BY
    no_of_emp DESC;



-------------------------------------------------------------------------------
-- CONVERSE FUNCTIONS ---------------------------------------------------------
/*
CAST(exp AS typename)
	typename		exp
	BOOL			INT64, BOOL	, STRING
	DATE, DATETIME 	STRING, TIME, DATETIME, TIMESTAMP
	FLOAT64, INT64 	INT64, FLOAT64, NUMERIC, BIGNUMERIC, STRING 	
	NUMERIC 		INT64, FLOAT64, NUMERIC, BIGNUMERIC, STRING 
	INTERVAL 		STRING
	STRING		 	....
*/
		
SELECT *,
  CAST(100*(monthly_change-1) AS INT) AS growth
FROM `nod-sql-copy.data_cleaning.orders`;



-------------------------------------------------------------------------------
-- CONDITIONAL EXPRESSIONS ----------------------------------------------------
/*
Evaluated left to right, with short-circuiting, and only evaluate the output 
value that was chosen

https://cloud.google.com/bigquery/docs/reference/standard-sql/conditional_expressions

CASE
	Syntax alt 1:
		CASE expr
			WHEN expr_to_match THEN result
			[ ... ]
			[ ELSE else_result ]
			END [AS name]
Syntax alt 2:
	CASE 
		WHEN condition THEN result
		[ ... ]
		[ ELSE else_result ]
		END [AS name]

COALESCE(expr[, ...])
	Returns the value of the first non-null expression
	COALESCE(A,B,C) returns B if A=Null and b1=Null

IF(expr, true_result, else_result)
	If expr is true, returns true_result, else returns else_result

IFNULL(expr, null_result)
	If expr is NULL, return null_result. Otherwise, return expr

NULLIF(expr, expr_to_match)
	Returns NULL if expr = expr_to_match is true, otherwise returns expr
*/

SELECT *
	CASE A
		WHEN 90 THEN 'red'
		WHEN 50 THEN 'blue'
		ELSE 'green'
		END AS result
FROM Numbers

SELECT COUNT(*),
	CASE 
		WHEN salary > (
			SELECT AVG(salary)
			FROM `nod-sql-copy.emp.salaries`
			)
			THEN "YES"
		ELSE "NO"
		END AS sal_above_avg
FROM `nod-sql-copy.emp.salaries`
GROUP BY sal_above_avg;

SELECT COALESCE(NULL, 'B', 'C') as result;

SELECT *,
	IF(A < B, 'true', 'false') AS result
FROM Numbers;



-------------------------------------------------------------------------------
-- JOIN -----------------------------------------------------------------------
/*
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#join_types

Cross join operation 
	Syntax:
		[INNER] JOIN | LEFT [OUTER] JOIN | RIGHT [OUTER] JOIN | FULL [OUTER] JOIN
		ON bool_expression | USING (col)
Comments:
	Common: [INNER] JOIN, LEFT [OUTER] JOIN
	ON return multiple columns while USING returns one
	Cheat sheet: joins_cheat_sheet.pdf

Condition join operation
	Syntax:
		ROSS JOIN
*/

SELECT *
FROM `nod-sql-copy.emp.employees` AS m
LEFT JOIN `nod-sql-copy.emp.salaries`AS s
ON m.emp_no=s.emp_no
WHERE s.salary IS NULL;

SELECT *
FROM `nod-sql-copy.emp.employees` AS m
LEFT JOIN `nod-sql-copy.emp.salaries`AS s
USING (emp_no);

SELECT *
FROM `nod-sql-copy.emp.salaries` AS s
JOIN `nod-sql-copy.emp.employees` AS e
  ON s.emp_no = e.emp_no
JOIN emp.titles AS t
  ON e.emp_no = t.emp_no
WHERE t.title = "Senior Engineer"
ORDER BY e.emp_no;



-------------------------------------------------------------------------------
-- SUBQUERY -------------------------------------------------------------------

-- In WHERE
SELECT *
FROM `nod-sql-copy.emp.employees`
WHERE emp_no IN (
  SELECT emp_no
  FROM `nod-sql-copy.emp.salaries`
  WHERE salary > (
    SELECT AVG(salary)
    FROM `nod-sql-copy.emp.salaries`
  )
);

-- In FROM
SELECT quarter,
  COUNT(*) AS new_emp_per_qu
FROM (SELECT *, 
  DATE_TRUNC(hire_date,QUARTER) AS quarter
  FROM `nod-sql-copy.emp.employees`
)
GROUP BY quarter
ORDER BY quarter;

-- In SELECT
-- Usefull in combination with CASE
SELECT *,
  (SELECT AVG(salary) AS averg_salary
  FROM `nod-sql-copy.emp.salaries` AS s
  WHERE s.emp_no = e.emp_no)
FROM `nod-sql-copy.emp.employees` AS e;



-------------------------------------------------------------------------------
-- WINDOW FUNCTIONS -----------------------------------------------------------
/*
https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts

An analytic function computes values over a group of rows and returns a single 
result for each row. This is different from an aggregate function, which returns 
a single result for a group of rows.

With analytic functions you can compute: 
	- moving averages, 
    - rank items, 
    - calculate cumulative sums, 
    - and lots more ...

Syntax
	analytical_func()
	OVER(
		{named_window | window_expression}
	) [AS name]
	...
	[WINDOW named_window AS (window_expression)] 

	window_expression:
		[named_window]
		[PARTITION BY partition_expression [, ...]]
		[ORDER BY expression [ { ASC | DESC } ] [, ...]]
		[window_frame_clause]

Example of analytical functions:
	RANK()
		Syntax:
		PARTITION BY: Optional.
		ORDER BY: Required, except for ROW_NUMBER().
		window_frame_clause: Disallowed.
	AVG(col)
	SUM(col)
	COUNT(* | col)
	LAG (value_expression[, offset [, default_expression]])		
		Preceding value
		offset: which subsequent row is returned; the default value is 1
	LEAD (value_expression[, offset [, default_expression]]) 		
		Next value
	FIRST_VALUE (value_expression [{RESPECT | IGNORE} NULLS])
	LAST_VALUE (value_expression [{RESPECT | IGNORE} NULLS])
	NTH_VALUE (value_expression, constant_integer_expression [{RESPECT | IGNORE} NULLS])
	PERCENTILE_CONT (value_expression, percentile [{RESPECT | IGNORE} NULLS])
		Computes the specified percentile value for the value_expression, with 
		linear interpolation
	PERCENTILE_DISC (value_expression, percentile [{RESPECT | IGNORE} NULLS])
	...
	For more functions: https://cloud.google.com/bigquery/docs/reference/standard-sql/mathematical_functions

*/

-- OVER + window_expression
SELECT *,
  AVG(meal_price) OVER() AS avg_price,
  AVG(meal_price) OVER(PARTITION BY eatery) AS avg_price_eatyer,
  RANK() OVER(ORDER BY meal_price) AS meal_rank,
  RANK() OVER(PARTITION BY eatery ORDER BY meal_price) AS meal_rank_eatery
FROM `nod-sql-copy.data_cleaning.meals`;

-- Moving average
SELECT *,
  AVG(order_count)
  OVER (
    ORDER BY order_date
    ROWS BETWEEN 7 PRECEDING AND 0 FOLLOWING
  ) AS avg_order_c
FROM `nod-sql-copy.data_cleaning.order_count`
ORDER BY order_date;

-- Cumulative Sum
SELECT *,
  SUM(order_count)
  OVER (
    ORDER BY year_month
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS cum_sum_orders
FROM `nod-sql-copy.data_cleaning.order_count`
ORDER BY year_month;

-- Growth Rate
SELECT *,
  order_count / LAG(order_count)
  OVER (
    ORDER BY year_month
  ) AS monthly_change
FROM `nod-sql-copy.data_cleaning.order_count`
ORDER BY year_month; 

-- OVER + WINDOW + named_window
SELECT LAST_VALUE(item) OVER (item_window) AS most_popular,
  LAST_VALUE(item) OVER (d) AS most_popular2
FROM Produce
WINDOW item_window AS (
  	PARTITION BY category
  	ORDER BY purchases
  	ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING),
  a AS (PARTITION BY category),
  b AS (a ORDER BY purchases),
  c AS (b ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING),
  d AS (c);



-------------------------------------------------------------------------------
-- JSON FUNCTIONS -------------------------------------------------------------



-------------------------------------------------------------------------------
-- READABILITY ----------------------------------------------------------------
/*
Code is read 10x more than it is written

Quick introduction to SQL formater
https://poorsql.com/
indent string: \s\s\s\s
max line width: 79
*/


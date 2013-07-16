import os
from django.db import connection


def execute(sql):
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    finally:
        cursor.close()
    return(rows)

def get_dataset(archive_id,variable_name,scenario_code,time_range,cm_code):
    sql = """
with
climatemodel as (
select *
from climatedata_climatemodel
where archive_id = {archive_id}  and 
      code = '{cm_code}'
),
variable as (
select *
from climatedata_variable
where name = '{variable_name}'
),
scenario as (
select *
from climatedata_scenario
where code = '{scenario_code}'
),
base_datasets as (
select *
from climatedata_dataset 
where climatemodel_id in (select id from climatemodel) and 
      scenario_id in (select id from scenario)
),
time as (
select distinct dataset_id
from climatedata_indextime
where dataset_id in (select id from base_datasets) and 
      value between '{lower}' and '{upper}'
)

select variable.dataset_id
from variable,time
where variable.dataset_id = time.dataset_id;
"""
    sql = sql.format(archive_id=archive_id,
                     variable_name=variable_name,
                     scenario_code=scenario_code,
                     lower=time_range[0],
                     upper=time_range[1],
                     cm_code=cm_code)
    return(sql)
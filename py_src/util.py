
def extract_marks(label_data, task, run) -> tuple | None:
    task, run = task, run
    query_exp = f'`Task Code (Task ID)` == @task and `Trial ID` == @run'  # extract correct row
    row = label_data.query(query_exp)
    if not row.empty: 
        onset_frame = row.iat[0, 3]
        impact_frame = row.iat[0, 4]
        return onset_frame, impact_frame
    return None
from src.sources.job_descriptions import measure_duplicates


def test_repeat_rows_on_the_key_and_on_all_columns_but_job_id(tmp_path):
    p = tmp_path / "jd.csv"
    p.write_text("Job Id,Job Title,Role,skills,Company,Job Posting Date,Other\n"
                 "1,Accountant,Analyst,excel,ACME,2023-01-02,x\n"
                 "2,Accountant,Analyst,excel,ACME,2023-01-02,y\n"
                 "3,Accountant,Analyst,excel,ACME,2023-01-02,y\n"
                 "4,Nurse,Clinician,care,Health Co,2023-01-09,z\n")
    r = measure_duplicates(p, chunk_rows=2)
    assert r["rows"] == 4
    assert r["repeat_rows_key"] == 2 and r["share_repeat_rows_key"] == 0.5
    assert r["repeat_rows_except_job_id"] == 1 and r["share_repeat_rows_except_job_id"] == 0.25
    assert r["key_columns"] == ["Job Title", "Role", "skills", "Company", "Job Posting Date"]
    assert r["key_columns_absent"] == []


def test_missing_key_columns_are_reported_not_fatal(tmp_path):
    p = tmp_path / "jd.csv"
    p.write_text("Job Id,Job Title,Role,skills,Job Posting Date\n1,A,B,c,2023-01-02\n2,A,B,c,2023-01-02\n")
    r = measure_duplicates(p)
    assert r["key_columns_absent"] == ["Company"]
    assert r["repeat_rows_key"] == 1

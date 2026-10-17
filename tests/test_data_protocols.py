import numpy as np
import pandas as pd
import pytest
from prototype_new.src.data.protocols import retail_windows,validate_retail,bounded_parquet


def fixture():
    return pd.DataFrame([dict(store_id=1,product_id=2,dt=str(d.date()),sale_amount=float(i),
        hours_sale=np.ones(24)*i/24,hours_stock_status=np.zeros(24),stock_hour6_22_cnt=0)
        for i,d in enumerate(pd.date_range('2024-01-01',periods=10))])


def test_retail_chronology_and_no_future_input():
    f=fixture(); w=retail_windows(f,2,'2024-01-05','2024-01-07')
    assert [len(w[s]) for s in w]==[3,2,3]
    for split,rows in w.items():
        for r in rows: assert r['x'][-1,0]==r['sales']-1
    changed=f.copy(); changed.loc[9,'sale_amount']=999
    other=retail_windows(changed,2,'2024-01-05','2024-01-07')
    np.testing.assert_array_equal(w['test'][-1]['x'],other['test'][-1]['x'])
    assert other['test'][-1]['sales']==999


def test_retail_rejects_duplicates_bad_hours_and_boundaries():
    f=fixture()
    with pytest.raises(ValueError): validate_retail(pd.concat([f,f.iloc[:1]]))
    f.at[0,'hours_sale']=np.zeros(23)
    with pytest.raises(ValueError): validate_retail(f)
    with pytest.raises(ValueError): retail_windows(fixture(),2,'2024-01-07','2024-01-05')


def test_bounded_projection(tmp_path):
    f=fixture(); f.to_parquet(tmp_path/'data.parquet')
    out=bounded_parquet(tmp_path/'data.parquet',['dt','sale_amount'],3)
    assert len(out)==3 and list(out)==['dt','sale_amount']


def test_stockout_count_matches_operating_hours():
    f=fixture(); f.at[0,'stock_hour6_22_cnt']=1
    with pytest.raises(ValueError,match='disagrees'):validate_retail(f)
    values=np.zeros(24);values[6]=1;f.at[0,'hours_stock_status']=values
    assert len(validate_retail(f))==10

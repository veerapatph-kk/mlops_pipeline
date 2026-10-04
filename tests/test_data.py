"""Data tests — ตรวจว่าข้อมูลยังหน้าตาเหมือนที่ตกลงไว้ ก่อนจะเอาไปเทรน"""
from sklearn.datasets import load_breast_cancer
 
df = load_breast_cancer(as_frame=True).frame
 
 
def test_schema():
    """คอลัมน์ต้องครบ 13 ฟีเจอร์ + target"""
    assert df.shape[1] == 31
    assert "target" in df.columns
 
 
def test_no_missing():
    assert df.isnull().sum().sum() == 0
 
 
def test_three_classes():
    assert df["target"].nunique() == 2
 
 
def test_alcohol_range():
    """ค่าที่หลุดช่วงนี้แปลว่าข้อมูลต้นทางผิดปกติ"""
    assert df["alcohol"].between(5.0, 30.0).all()

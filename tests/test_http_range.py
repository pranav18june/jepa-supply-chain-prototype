import pytest
from prototype_new.src.data.http_range import HTTPRangeReader


def test_bounded_range_seek_cache_and_budget(monkeypatch):
    content=b'0123456789abcdef'; calls=[]
    class Response:
        status=206
        def __init__(self,start,end):
            self.start,self.end=start,end;self.headers={'Content-Range':f'bytes {start}-{end}/16'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return content[self.start:self.end+1]
    def open_request(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'));calls.append((start,end));return Response(start,end)
    monkeypatch.setattr('urllib.request.urlopen',open_request)
    reader=HTTPRangeReader('https://example.test/data',16,block_size=4,max_transfer=12)
    assert reader.read(6)==b'012345'
    reader.seek(-2,2);assert reader.read()==b'ef'
    reader.seek(0);assert reader.read(4)==b'0123'
    assert len(calls)==3
    reader.seek(8)
    with pytest.raises(IOError,match='budget'):reader.read(4)

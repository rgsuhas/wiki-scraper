import pytest
from src.wikipedia_scraper.wikipedia_scraper import fetch_wikipedia_html, parse_wikipedia, build_person, parse_infobox

class DummyResponse:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")
        

def test_fetch_success(monkeypatch):
    mock_html = "<html><h1 id='firstHeading'>Test Person</h1><p>Bio here</p></html>"
    monkeypatch.setattr("requests.get", lambda url, timeout=5: DummyResponse(mock_html))
    html = fetch_wikipedia_html("Test_Person")
    assert "Test Person" in html

def test_fetch_404_error(monkeypatch):
    def mock_get_404(url, timeout=5):
        response = DummyResponse("", 404)
        response.raise_for_status = lambda: Exception("404 Not Found") # type: ignore
        return response
    
    monkeypatch.setattr("requests.get", mock_get_404)
    html = fetch_wikipedia_html("Nonexistent_Person")
    assert html == ""

def test_parse_wikipedia():
    html = "<html><h1 id='firstHeading'>Test Person</h1><p>Bio here</p></html>"
    name, bio = parse_wikipedia(html)
    assert name == "Test Person"
    assert bio == "Bio here"

def test_parse_wikipedia_no_title():
    html = "<html><p>Bio here</p></html>"
    name, bio = parse_wikipedia(html)
    assert name == ""
    assert bio == "Title not found"

def test_parse_infobox():
    html = """
    <html>
      <table class='infobox'>
        <tr><th>Born</th><td>1900</td></tr>
        <tr><th>Occupation</th><td>Actor, Singer</td></tr>
      </table>
    </html>
    """
    data = parse_infobox(html)
    assert data["Born"] == "1900"
    assert data["Occupation"] == "Actor, Singer"

def test_parse_infobox_empty():
    html = "<html><p>No infobox here</p></html>"
    data = parse_infobox(html)
    assert data == {}

def test_build_person():
    html = """
    <html>
      <h1 id='firstHeading'>Test Person</h1>
      <p>Bio here</p>
      <table class='infobox'>
        <tr><th>Born</th><td>1900</td></tr>
        <tr><th>Occupation</th><td>Actor, Singer</td></tr>
      </table>
    </html>
    """
    person = build_person(html, "Test_Person")
    assert person.full_name == "Test Person"
    assert person.birth == "1900"
    assert "Actor" in person.occupation
    assert person.bio == "Bio here"

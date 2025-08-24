import argparse
import requests
from bs4 import BeautifulSoup, Tag  # <-- Add Tag import
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


def fetch_wikipedia_html(name: str) -> str:
    url = f"https://en.wikipedia.org/wiki/{name}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as e:
        if response.status_code == 404:
            print(f"Person '{name}' not found on Wikipedia")
        else:
            print(f"HTTP error: {e}")
        return ""
    except requests.RequestException as e:
        print(f"Network error: {e}")
        return ""


def parse_wikipedia(html: str):
    soup = BeautifulSoup(html, "html.parser")
    
    # Full name from page title
    title_element = soup.find("h1", id="firstHeading")
    if not title_element:
        return "", "Title not found"
    
    full_name = title_element.text.strip()
    
    # First paragraph of biography (skip empty or table text)
    first_para = ""
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text and len(text) > 1: # skip empty bios
            first_para = text
            break
    
    return full_name, first_para


def parse_infobox(html: str):
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.find("table", class_="infobox")
    if not infobox:
        return {}
    
    data = {}
    for row in infobox.find_all("tr"):
        header = row.find("th")
        value = row.find("td")
        if header and value:
            header_text = header.get_text(strip=True)
            value_text = value.get_text(" ", strip=True)
            if header_text and value_text:
                data[header_text] = value_text
    
    # Debug: print what we found
    print(f"DEBUG: Found infobox fields: {list(data.keys())}")
    
    return data


@dataclass
class Person:
    full_name: str
    birth: Optional[str] = None
    death: Optional[str] = None
    occupation: List[str] = field(default_factory=list)
    bio: Optional[str] = None

class InfoType(Enum):
    FULL_NAME = "Full Name"
    BIRTH = "Birth"
    DEATH = "Death"
    OCCUPATION = "Occupation"
    BIO = "Biography"


def build_person(html: str, name: str) -> Person:
    full_name, bio = parse_wikipedia(html)
    infobox = parse_infobox(html)

    occupations = []
    # Try different possible occupation field names
    occupation_fields = ["Occupation", "Occupations", "Profession", "Professions", "Work", "Career"]
    
    for field in occupation_fields:
        if field in infobox:
            occupation_text = infobox[field]
            # Handle different separators and clean up
            cleaned_text = occupation_text.replace(" and ", ",").replace("&", ",")
            # If no commas, try to split by common patterns
            if "," not in cleaned_text:
                # Split by common occupation separators
                cleaned_text = cleaned_text.replace("  ", " ").replace(" ", ",")
            
            # Split by comma and clean each occupation
            occupations = [o.strip() for o in cleaned_text.split(",") if o.strip()]
            break
    
    return Person(
        full_name=full_name,
        birth=infobox.get("Born"),
        death=infobox.get("Died"),
        occupation=occupations,
        bio=bio
    )



def main():
    parser = argparse.ArgumentParser(description="Fetch person info from Wikipedia")
    parser.add_argument("--name", required=True, help="Person's name (use underscores for spaces)")
    args = parser.parse_args()

    print(f"Fetching information for: {args.name}")
    
    html_content = fetch_wikipedia_html(args.name)
    if not html_content:
        print("Failed to fetch Wikipedia page. Please check the name and try again.")
        return

    try:
        person = build_person(html_content, args.name)
        
        print("\n" + "="*50)
        print(f" {person.full_name}")
        print("="*50)
        
        if person.birth:
            print(f" Born: {person.birth}")
        if person.death:
            print(f" Died: {person.death}")
        if person.occupation:
            print(f" Occupation: {', '.join(person.occupation)}")
        if person.bio:
            print(f"\n Biography:")
            print(f"   {person.bio[:200]}{'...' if len(person.bio) > 200 else ''}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error processing data: {e}")


if __name__ == "__main__":
    main()

---

## **Section 1: Linux Setup and Environment** (1.5h)

**Tasks:**

1. **Read**: [Why GNU/Linux matters](https://www.gnu.org/gnu/why-gnu-linux.en.html).✅
2. **Decide Install Method**: ✅

   * Dual Boot
   * Virtual Machine (VirtualBox)✅
3. **If Dual Boot**:✅

   * Backup all data.
   * Create bootable USBs for Ubuntu/Fedora.
   * Partition disk: 25GB minimum, 4–8GB swap.
   * Install Ubuntu → Fedora (if both).
   * Configure GRUB bootloader.
4. **If Virtual Machine**:

   * Download and install VirtualBox.
   * Allocate 4–8GB RAM, 40GB storage.
   * Install chosen Linux distribution.
   * Install Guest Additions.
5. **Post-Installation Setup**:

   * Update system:✅

     ```bash
     sudo apt update && sudo apt upgrade
     ```
   * Install development tools:✅

     ```bash
     sudo apt install python3-dev python3-pip python3-venv git curl wget vim build-essential
     ```
   * Configure Git with name and email.✅

---

## **Section 2: Linux Command Line Mastery** (2h)

**Tasks:**

1. **Learn** Linux commands using [Linux Journey](https://linuxjourney.com/).
2. **Understand** filesystem structure: `/`, `/home`, `/etc`, `/var`, `/usr`, `/opt`, `/tmp`.
3. **Practice File Permissions**:

   - `chmod 755 script.py`
   - `chmod 600 private_key`
   - `ls -la`

4. **Package Management**:

   - Ubuntu/Debian: `sudo apt install package-name`
   - Fedora/Red Hat: `sudo dnf install package-name`

5. **User Management**:

   - `whoami`
   - `sudo useradd -m -s /bin/bash newuser`
   - `sudo usermod -aG sudo newuser`

6. **Exercises**:

   - Create project folders with `mkdir -p`.
   - Edit files, run `ps aux`, `htop`, and manage background processes.

---

## **Section 3: VS Code IDE Setup** (30m)

**Tasks:**

1. Download and install VS Code.
2. Install Python extension.
3. Configure Python interpreter (Ctrl+Shift+P → “Python: Select Interpreter”).
4. Enable auto-formatting and linting.

---

## **Section 4: Python Environment Management** (30m)

**Tasks:**

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate:

   ```bash
   source .venv/bin/activate
   ```

3. Install packages:

   ```bash
   pip install requests numpy
   ```

4. Freeze dependencies: `pip freeze > requirements.txt`
5. Add `.venv` to `.gitignore`.

---

## **Section 5: Dependency Management with Hatch** (1h)

**Tasks:**

1. Install Hatch: `pip install hatch`.
2. Initialize project:

   ```bash
   hatch new my-project
   ```

3. Add dependencies in `pyproject.toml`.
4. Create and manage multiple environments:

   ```bash
   hatch env create test
   hatch run test:pytest
   ```

5. Run linting and formatting with Hatch scripts.

---

## **Section 6: Advanced Python Concepts** (3h)

**Tasks:**

1. Review Python type hints and exception handling.
2. Implement OOP example (BankAccount).
3. Practice `@dataclass` and `Enum` usage.
4. Build combined exercise: Task Management System.

---

## **Section 7: Unit Testing Best Practices** (2h)

**Tasks:**

1. Write tests using `unittest`.
2. Write tests using `pytest`.
3. Practice parametrized tests.
4. Follow AAA pattern (Arrange-Act-Assert).
5. Organize tests in `tests/` directory.

---

## **Section 8: Code Patterns and Design Patterns** (2h)

**Tasks:**

1. Implement Interface pattern using `ABC`.
2. Implement Factory pattern.
3. Implement Singleton and Observer patterns.
4. Apply SOLID principles.
5. Build Notification System combining patterns.

---

## **Final Project: Wikipedia Person Information Extractor** (2–3h)

**Tasks:**

1. Accept person name as CLI argument (`argparse`).
2. Fetch Wikipedia page (`requests`).
3. Parse HTML (`BeautifulSoup`) for:

   - Full name
   - Birth date and place
   - Death date (if applicable)
   - Occupation(s)
   - First paragraph of biography

4. Store data in `@dataclass`.
5. Use `Enum` for information types.
6. Add error handling.
7. Write unit tests with mocked responses.
8. Follow design patterns from earlier sections.

---

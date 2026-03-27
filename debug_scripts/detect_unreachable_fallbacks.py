#!/usr/bin/env python
"""Detect unreachable fallback patterns in conditional return statements.

Uses astroid for semantic control flow analysis. Identifies functions where
fallback code (typically final return or else block) is unreachable because
preceding conditions are exhaustive.

Example patterns detected:
    if condition1:
        return value1
    elif condition2:
        return value2
    else:
        return DEFAULT  # ← Unreachable if condition1 + condition2 are exhaustive

Uses astroid (installed: 4.0.4+) instead of built-in ast for better control
flow and semantic analysis.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import astroid


@dataclass
class UnreachableFallback:
    """Report of an unreachable fallback pattern."""

    filepath: str
    function_name: str
    line_no: int
    fallback_type: str  # "final_return", "else_block", "except_handler"
    description: str
    code_snippet: str


class FallbackDetector:
    """Detect unreachable fallback patterns using astroid semantic analysis."""

    def __init__(self, filepath: str, source_lines: List[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.findings: List[UnreachableFallback] = []

    def detect(self, module: astroid.Module) -> List[UnreachableFallback]:
        """Scan module for unreachable fallback patterns."""
        for node in module.nodes_of_class(astroid.FunctionDef, astroid.AsyncFunctionDef):
            self._check_function_body(node)
        return self.findings

    def _check_function_body(self, func_node: astroid.FunctionDef):
        """Check function body for unreachable fallback patterns."""
        body = func_node.body
        if not body:
            return

        # Pattern 1: Final return after exhaustive if/elif chain
        self._check_exhaustive_if_chain(body, func_node)

        # Pattern 2: Try/except with redundant finally return
        self._check_try_except_finally(body, func_node)

        # Pattern 3: Guarded returns with questionable fallback
        self._check_guarded_return_chain(body, func_node)

    def _check_exhaustive_if_chain(self, body: List, func_node):
        """Detect exhaustive if/elif chains with unreachable final returns."""
        for stmt in body:
            if not isinstance(stmt, astroid.If):
                continue

            # Collect if/elif chain
            if_chain = self._collect_if_elif_chain(stmt)
            if len(if_chain) < 2:  # Need at least if + elif/else
                continue

            # Check if final chain item has an else
            last_if = if_chain[-1]
            has_final_else = bool(last_if.orelse)

            if not has_final_else:
                continue

            # Check if all branches return
            all_branches_return = all(self._branch_always_returns(s.body) for s in if_chain)

            if all_branches_return and has_final_else:
                else_body = last_if.orelse
                if len(else_body) == 1 and isinstance(else_body[0], astroid.Return):
                    snippet = self._get_code_snippet(last_if.lineno, last_if.end_lineno)
                    self.findings.append(
                        UnreachableFallback(
                            filepath=self.filepath,
                            function_name=func_node.name,
                            line_no=else_body[0].lineno,
                            fallback_type="final_return",
                            description=(
                                f"Unreachable return in else block after exhaustive "
                                f"if/elif chain ({len(if_chain)} branches all return)"
                            ),
                            code_snippet=snippet,
                        )
                    )

    def _check_try_except_finally(self, body: List, func_node):
        """Detect try/except with redundant finally return."""
        for stmt in body:
            if not isinstance(stmt, astroid.Try):
                continue

            if not stmt.finalbody:
                continue

            if len(stmt.finalbody) != 1 or not isinstance(stmt.finalbody[0], astroid.Return):
                continue

            # Check if all paths already return
            try_returns = self._branch_always_returns(stmt.body)
            handlers_return = all(self._branch_always_returns(h.body) for h in stmt.handlers)

            if try_returns and handlers_return:
                snippet = self._get_code_snippet(stmt.lineno, stmt.end_lineno)
                self.findings.append(
                    UnreachableFallback(
                        filepath=self.filepath,
                        function_name=func_node.name,
                        line_no=stmt.finalbody[0].lineno,
                        fallback_type="finally_return",
                        description=(
                            "Unreachable return in finally block "
                            "(all try/except branches return)"
                        ),
                        code_snippet=snippet,
                    )
                )

    def _collect_if_elif_chain(self, if_node: astroid.If) -> List[astroid.If]:
        """Collect all If nodes in an if/elif/elif... chain."""
        chain = [if_node]
        current = if_node

        while current.orelse and len(current.orelse) == 1:
            next_stmt = current.orelse[0]
            if isinstance(next_stmt, astroid.If):
                chain.append(next_stmt)
                current = next_stmt
            else:
                break

        return chain

    def _branch_always_returns(self, body: List) -> bool:
        """Check if a code block always returns/raises."""
        if not body:
            return False

        # Check last statement
        last_stmt = body[-1]
        if isinstance(last_stmt, (astroid.Return, astroid.Raise)):
            return True

        # Check if last statement is an if that always returns all branches
        if isinstance(last_stmt, astroid.If):
            if_chain = self._collect_if_elif_chain(last_stmt)
            if not last_stmt.orelse:
                return False  # No else, some paths don't return

            return all(
                self._branch_always_returns(s.body) for s in if_chain
            ) and self._branch_always_returns(last_stmt.orelse)

        return False

    def _get_code_snippet(self, start_line: int, end_line: int) -> str:
        """Extract code snippet from source."""
        if start_line is None or end_line is None:
            return "(snippet unavailable)"

        start_idx = start_line - 1
        end_idx = min(end_line, len(self.source_lines))

        lines = self.source_lines[start_idx:end_idx]
        return "\n".join(lines).strip()

    def _check_guarded_return_chain(self, body: List, func_node):
        """Detect multiple guarded returns followed by an unconditional fallback.

        Patterns like:
            if guard1:
                return value1
            if guard2:
                return value2
            return DEFAULT  # Fallback

        If guards are exhaustive (upstream validates), fallback is unreachable.
        """
        if len(body) < 3:
            return

        # Find sequence of if-returns (no orelse)
        consecutive_if_returns = []
        for i, stmt in enumerate(body):
            if (
                isinstance(stmt, astroid.If)
                and not stmt.orelse
                and len(stmt.body) == 1
                and isinstance(stmt.body[0], astroid.Return)
            ):
                consecutive_if_returns.append((i, stmt))
            else:
                # Stop collecting when we hit non-matching statement
                if consecutive_if_returns:
                    break

        # Check if we have 2+ guarded returns followed by a final return
        if len(consecutive_if_returns) >= 2:
            last_idx, last_if = consecutive_if_returns[-1]
            # Check if the very next statement is a return
            if last_idx + 1 < len(body):
                next_stmt = body[last_idx + 1]
                if isinstance(next_stmt, astroid.Return):
                    snippet = self._get_code_snippet(
                        consecutive_if_returns[0][1].lineno, next_stmt.lineno
                    )
                    self.findings.append(
                        UnreachableFallback(
                            filepath=self.filepath,
                            function_name=func_node.name,
                            line_no=next_stmt.lineno,
                            fallback_type="fallback_return",
                            description=(
                                f"Fallback return after {len(consecutive_if_returns)} "
                                f"guarded returns; likely unreachable if guards are exhaustive"
                            ),
                            code_snippet=snippet,
                        )
                    )


def scan_directory(dirpath: Path) -> List[UnreachableFallback]:
    """Scan a directory for Python files and detect unreachable fallbacks."""
    findings = []

    for pyfile in dirpath.rglob("*.py"):
        if any(part.startswith(".") for part in pyfile.parts):
            continue  # Skip hidden directories

        try:
            source = pyfile.read_text(encoding="utf-8")
            source_lines = source.splitlines()

            # Use astroid to parse (better semantic analysis)
            module = astroid.parse(source, module_name=pyfile.stem)

            detector = FallbackDetector(str(pyfile), source_lines)
            findings.extend(detector.detect(module))

        except (astroid.AstroidError, UnicodeDecodeError, SyntaxError) as e:
            print(f"Warning: Could not parse {pyfile}: {e}", file=sys.stderr)

    return findings


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect unreachable fallback patterns in Python source."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="kl_clustering_analysis",
        help="Directory to scan (relative to repo root or absolute). "
        "Default: kl_clustering_analysis",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    target_dir = Path(args.target)
    if not target_dir.is_absolute():
        target_dir = repo_root / target_dir

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {target_dir.relative_to(repo_root)}/ for unreachable fallback patterns...\n")

    findings = scan_directory(target_dir)

    if not findings:
        print("✅ No unreachable fallbacks detected!")
        return 0

    print(f"⚠️  Found {len(findings)} potential unreachable fallback(s):\n")

    for i, finding in enumerate(findings, 1):
        try:
            rel_path = Path(finding.filepath).relative_to(repo_root).as_posix()
        except ValueError:
            rel_path = finding.filepath
        print(f"{i}. {rel_path}:{finding.line_no} in `{finding.function_name}()`")
        print(f"   Type: {finding.fallback_type}")
        print(f"   Issue: {finding.description}")
        print(
            f"   Code:\n{chr(10).join('     ' + line for line in finding.code_snippet.split(chr(10)))}"
        )
        print()

    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())

"""
LanceDB CLI - Manage jobs, resumes, and matches in LanceDB database

Usage:
    python -m graph.lancedb_cli <command> [options]
    
Commands:
    list-jobs       List all jobs in database
    list-resumes    List all resumes in database
    list-matches    List all matches in database
    delete-jobs     Delete all jobs
    delete-resumes  Delete all resumes
    delete-matches  Delete all matches
    clear-all       Clear entire database
    stats           Show database statistics
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from graph.database import graphDB
from graph.settings import get_settings

console = Console()


class LanceDBManager:
    """Manage LanceDB database operations"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager"""
        try:
            self.db = graphDB(db_path=db_path, reset_if_exists=False)
            self.settings = get_settings()
        except Exception as e:
            console.print(f"[red]Error connecting to database: {e}[/red]")
            sys.exit(1)
    
    def list_jobs(self, limit: int = 20):
        """List jobs in database"""
        try:
            if 'jobs' not in self.db.db.table_names():
                console.print("[yellow]No jobs table found[/yellow]")
                return
            
            table = self.db.db.open_table('jobs')
            df = table.to_pandas()
            
            if df.empty:
                console.print("[yellow]No jobs found in database[/yellow]")
                return
            
            # Create rich table
            rich_table = Table(title=f"Jobs in Database (showing {min(limit, len(df))} of {len(df)})")
            rich_table.add_column("Job ID", style="cyan", no_wrap=True)
            rich_table.add_column("Title", style="green")
            rich_table.add_column("Company", style="yellow")
            rich_table.add_column("Location", style="blue")
            rich_table.add_column("Query", style="magenta")
            
            for _, row in df.head(limit).iterrows():
                title = str(row.get('title', 'Unknown'))[:40]
                company = str(row.get('company', 'Unknown'))[:30]
                location = str(row.get('location', row.get('search_location', 'Unknown')))[:25]
                
                rich_table.add_row(
                    str(row['job_id'])[:20] + '...' if len(str(row['job_id'])) > 20 else str(row['job_id']),
                    title + '...' if len(title) == 40 else title,
                    company + '...' if len(company) == 30 else company,
                    location + '...' if len(location) == 25 else location,
                    str(row.get('search_query', 'Unknown'))
                )
            
            console.print(rich_table)
            
        except Exception as e:
            console.print(f"[red]Error listing jobs: {e}[/red]")
    
    def list_resumes(self, limit: int = 20):
        """List resumes in database"""
        try:
            if 'resumes' not in self.db.db.table_names():
                console.print("[yellow]No resumes table found[/yellow]")
                return
            
            table = self.db.db.open_table('resumes')
            df = table.to_pandas()
            
            if df.empty:
                console.print("[yellow]No resumes found in database[/yellow]")
                return
            
            # Create rich table
            rich_table = Table(title=f"Resumes in Database (showing {min(limit, len(df))} of {len(df)})")
            rich_table.add_column("Resume ID", style="cyan", no_wrap=True)
            rich_table.add_column("Name", style="green")
            rich_table.add_column("Email", style="yellow")
            rich_table.add_column("Filename", style="blue")
            
            for _, row in df.head(limit).iterrows():
                rich_table.add_row(
                    str(row['resume_id'])[:20] + '...' if len(str(row['resume_id'])) > 20 else str(row['resume_id']),
                    str(row.get('name', 'Unknown')),
                    str(row.get('email', 'Unknown')),
                    str(row.get('filename', 'Unknown'))
                )
            
            console.print(rich_table)
            
        except Exception as e:
            console.print(f"[red]Error listing resumes: {e}[/red]")
    
    def list_matches(self, limit: int = 20):
        """List matches in database"""
        try:
            if 'matches' not in self.db.db.table_names():
                console.print("[yellow]No matches table found[/yellow]")
                return
            
            table = self.db.db.open_table('matches')
            df = table.to_pandas()
            
            if df.empty:
                console.print("[yellow]No matches found in database[/yellow]")
                return
            
            # Create rich table
            rich_table = Table(title=f"Matches in Database (showing {min(limit, len(df))} of {len(df)})")
            rich_table.add_column("Match ID", style="cyan", no_wrap=True)
            rich_table.add_column("Overall Score", style="green")
            rich_table.add_column("Semantic", style="yellow")
            rich_table.add_column("Skills", style="blue")
            rich_table.add_column("Experience", style="magenta")
            
            for _, row in df.head(limit).iterrows():
                rich_table.add_row(
                    str(row['match_id'])[:20] + '...' if len(str(row['match_id'])) > 20 else str(row['match_id']),
                    f"{row.get('overall_score', 0):.2%}",
                    f"{row.get('semantic_score', 0):.2%}",
                    f"{row.get('skills_score', 0):.2%}",
                    f"{row.get('experience_score', 0):.2%}"
                )
            
            console.print(rich_table)
            
        except Exception as e:
            console.print(f"[red]Error listing matches: {e}[/red]")
    
    def delete_all_jobs(self, confirm: bool = False, cascade: bool = True):
        """Delete all jobs from database
        
        Args:
            confirm: Require confirmation flag
            cascade: Also delete related matches (default: True)
        """
        if not confirm:
            console.print("[yellow]Use --confirm flag to delete all jobs[/yellow]")
            return
        
        try:
            jobs_deleted = 0
            matches_deleted = 0
            
            # Delete jobs
            if 'jobs' in self.db.db.table_names():
                # Get the table and count records
                table = self.db.db.open_table('jobs')
                df = table.to_pandas()
                jobs_deleted = len(df)
                
                if jobs_deleted == 0:
                    console.print("[yellow]No jobs to delete[/yellow]")
                    return
                
                # Drop and recreate the table to clear all data
                self.db.drop_table('jobs')
                
            # Delete related matches if cascade is enabled
            if cascade and 'matches' in self.db.db.table_names():
                table = self.db.db.open_table('matches')
                df = table.to_pandas()
                matches_deleted = len(df)
                
                if matches_deleted > 0:
                    self.db.drop_table('matches')
                    console.print(f"[yellow]  → Cascade deleted {matches_deleted} related matches[/yellow]")
            
            # Reinitialize tables
            self.db._init_tables()
            
            console.print(f"[green]Deleted {jobs_deleted} jobs successfully[/green]")
            if not cascade and 'matches' in self.db.db.table_names():
                table = self.db.db.open_table('matches')
                orphaned = len(table.to_pandas())
                if orphaned > 0:
                    console.print(f"[yellow]⚠ Warning: {orphaned} matches are now orphaned (use --cascade to delete them)[/yellow]")
                    
        except Exception as e:
            console.print(f"[red]Error deleting jobs: {e}[/red]")
    
    def delete_all_resumes(self, confirm: bool = False, cascade: bool = True):
        """Delete all resumes from database
        
        Args:
            confirm: Require confirmation flag  
            cascade: Also delete related matches (default: True)
        """
        if not confirm:
            console.print("[yellow]Use --confirm flag to delete all resumes[/yellow]")
            return
        
        try:
            resumes_deleted = 0
            matches_deleted = 0
            
            # Delete resumes
            if 'resumes' in self.db.db.table_names():
                # Get the table and count records
                table = self.db.db.open_table('resumes')
                df = table.to_pandas()
                resumes_deleted = len(df)
                
                if resumes_deleted == 0:
                    console.print("[yellow]No resumes to delete[/yellow]")
                    return
                
                # Drop and recreate the table to clear all data
                self.db.drop_table('resumes')
                
            # Delete related matches if cascade is enabled
            if cascade and 'matches' in self.db.db.table_names():
                table = self.db.db.open_table('matches')
                df = table.to_pandas()
                matches_deleted = len(df)
                
                if matches_deleted > 0:
                    self.db.drop_table('matches')
                    console.print(f"[yellow]  → Cascade deleted {matches_deleted} related matches[/yellow]")
            
            # Reinitialize tables
            self.db._init_tables()
            
            console.print(f"[green]Deleted {resumes_deleted} resumes successfully[/green]")
            if not cascade and 'matches' in self.db.db.table_names():
                table = self.db.db.open_table('matches')
                orphaned = len(table.to_pandas())
                if orphaned > 0:
                    console.print(f"[yellow]⚠ Warning: {orphaned} matches are now orphaned (use --cascade to delete them)[/yellow]")
                    
        except Exception as e:
            console.print(f"[red]Error deleting resumes: {e}[/red]")
    
    def delete_all_matches(self, confirm: bool = False):
        """Delete all matches from database"""
        if not confirm:
            console.print("[yellow]Use --confirm flag to delete all matches[/yellow]")
            return
        
        try:
            if 'matches' in self.db.db.table_names():
                # Get the table and count records
                table = self.db.db.open_table('matches')
                df = table.to_pandas()
                count = len(df)
                
                if count == 0:
                    console.print("[yellow]No matches to delete[/yellow]")
                    return
                
                # Drop and recreate the table to clear all data
                self.db.drop_table('matches')
                self.db._init_tables()
                console.print(f"[green]Deleted {count} matches successfully[/green]")
            else:
                console.print("[yellow]No matches table found[/yellow]")
        except Exception as e:
            console.print(f"[red]Error deleting matches: {e}[/red]")
    
    def clear_all(self, confirm: bool = False):
        """Clear all data from database without deleting the database itself"""
        if not confirm:
            console.print("[yellow]Use --confirm flag to clear entire database[/yellow]")
            console.print("[yellow]This will delete all jobs, resumes, and matches[/yellow]")
            return
        
        try:
            tables = self.db.db.table_names()
            total_deleted = 0
            
            for table_name in tables:
                try:
                    table = self.db.db.open_table(table_name)
                    df = table.to_pandas()
                    count = len(df)
                    
                    if count > 0:
                        # Drop and recreate each table
                        self.db.drop_table(table_name)
                        total_deleted += count
                        console.print(f"[green]  Cleared {count} records from {table_name}[/green]")
                except Exception as e:
                    console.print(f"[yellow]  ⚠ Could not clear {table_name}: {e}[/yellow]")
            
            # Reinitialize all tables
            self.db._init_tables()
            
            console.print(f"[bold green]Database cleared successfully![/bold green]")
            console.print(f"[green]  Total records deleted: {total_deleted}[/green]")
            console.print(f"[dim]  Database structure preserved at: {self.db.db_path}[/dim]")
        except Exception as e:
            console.print(f"[red]Error clearing database: {e}[/red]")
    
    def show_stats(self):
        """Show database statistics"""
        try:
            stats = {}
            tables = self.db.db.table_names()
            
            for table_name in tables:
                table = self.db.db.open_table(table_name)
                count = len(table.to_pandas())
                stats[table_name] = count
            
            # Create stats panel
            stats_text = ""
            total_records = 0
            for table, count in stats.items():
                stats_text += f"[cyan]{table}:[/cyan] {count:,} records\n"
                total_records += count
            
            stats_text += f"\n[bold]Total:[/bold] {total_records:,} records"
            stats_text += f"\n[bold]Database Path:[/bold] {self.db.db_path}"
            
            panel = Panel(
                stats_text,
                title="Database Statistics",
                border_style="green"
            )
            
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LanceDB Database Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List commands
    list_jobs = subparsers.add_parser('list-jobs', help='List all jobs')
    list_jobs.add_argument('--limit', type=int, default=20, help='Number of jobs to show')
    
    list_resumes = subparsers.add_parser('list-resumes', help='List all resumes')
    list_resumes.add_argument('--limit', type=int, default=20, help='Number of resumes to show')
    
    list_matches = subparsers.add_parser('list-matches', help='List all matches')
    list_matches.add_argument('--limit', type=int, default=20, help='Number of matches to show')
    
    # Delete commands
    delete_jobs = subparsers.add_parser('delete-jobs', help='Delete all jobs')
    delete_jobs.add_argument('--confirm', action='store_true', help='Confirm deletion')
    delete_jobs.add_argument('--no-cascade', action='store_true', help='Do not delete related matches')
    
    delete_resumes = subparsers.add_parser('delete-resumes', help='Delete all resumes')
    delete_resumes.add_argument('--confirm', action='store_true', help='Confirm deletion')
    delete_resumes.add_argument('--no-cascade', action='store_true', help='Do not delete related matches')
    
    delete_matches = subparsers.add_parser('delete-matches', help='Delete all matches')
    delete_matches.add_argument('--confirm', action='store_true', help='Confirm deletion')
    
    # Clear all
    clear_all = subparsers.add_parser('clear-all', help='Clear entire database')
    clear_all.add_argument('--confirm', action='store_true', help='Confirm clearing')
    
    # Stats
    stats = subparsers.add_parser('stats', help='Show database statistics')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = LanceDBManager()
    
    # Execute command
    if args.command == 'list-jobs':
        manager.list_jobs(args.limit)
    elif args.command == 'list-resumes':
        manager.list_resumes(args.limit)
    elif args.command == 'list-matches':
        manager.list_matches(args.limit)
    elif args.command == 'delete-jobs':
        cascade = not args.no_cascade if hasattr(args, 'no_cascade') else True
        manager.delete_all_jobs(args.confirm, cascade=cascade)
    elif args.command == 'delete-resumes':
        cascade = not args.no_cascade if hasattr(args, 'no_cascade') else True
        manager.delete_all_resumes(args.confirm, cascade=cascade)
    elif args.command == 'delete-matches':
        manager.delete_all_matches(args.confirm)
    elif args.command == 'clear-all':
        manager.clear_all(args.confirm)
    elif args.command == 'stats':
        manager.show_stats()


if __name__ == '__main__':
    main()

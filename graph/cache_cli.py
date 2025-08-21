"""
Cache management CLI for the job matching pipeline
"""

import argparse
import json
from pathlib import Path
from graph.cache_manager import CacheManager


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Manage cache for the job matching pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Cache commands")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument(
        "--cache-dir",
        default="data/.cache",
        help="Cache directory (default: data/.cache)"
    )
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument(
        "--cache-dir",
        default="data/.cache",
        help="Cache directory (default: data/.cache)"
    )
    clear_parser.add_argument(
        "--type",
        choices=["jobs", "resumes", "matches", "all"],
        default="all",
        help="Type of cache to clear (default: all)"
    )
    
    # Invalidate command
    invalidate_parser = subparsers.add_parser(
        "invalidate", 
        help="Remove expired cache entries"
    )
    invalidate_parser.add_argument(
        "--cache-dir",
        default="data/.cache",
        help="Cache directory (default: data/.cache)"
    )
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show cache details")
    show_parser.add_argument(
        "--cache-dir",
        default="data/.cache",
        help="Cache directory (default: data/.cache)"
    )
    show_parser.add_argument(
        "--type",
        choices=["jobs", "resumes", "matches"],
        required=True,
        help="Type of cache to show"
    )
    show_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache manager
    cache_manager = CacheManager(args.cache_dir)
    
    if args.command == "stats":
        stats = cache_manager.get_cache_stats()
        print("\n📊 Cache Statistics")
        print("=" * 50)
        print(f"\n📁 Jobs Cache:")
        print(f"  Count: {stats['jobs']['count']}")
        if stats['jobs']['queries']:
            print("  Recent queries:")
            for query in stats['jobs']['queries'][:5]:
                parts = query.split("|")
                print(f"    - Query: '{parts[0]}' | Location: {parts[1]} | Max: {parts[2]}")
        
        print(f"\n📄 Resumes Cache:")
        print(f"  Count: {stats['resumes']['count']}")
        if stats['resumes']['files']:
            print("  Cached files:")
            for file in stats['resumes']['files'][:5]:
                print(f"    - {file}")
        
        print(f"\n🔗 Matches Cache:")
        print(f"  Count: {stats['matches']['count']}")
        
    elif args.command == "clear":
        if args.type == "all":
            cache_manager.clear_cache()
            print("✅ Cleared all cache")
        else:
            cache_manager.clear_cache(args.type)
            print(f"✅ Cleared {args.type} cache")
    
    elif args.command == "invalidate":
        cache_manager.invalidate_expired()
        print("✅ Removed expired cache entries")
        
        # Show updated stats
        stats = cache_manager.get_cache_stats()
        print(f"\nRemaining cache entries:")
        print(f"  Jobs: {stats['jobs']['count']}")
        print(f"  Resumes: {stats['resumes']['count']}")
        print(f"  Matches: {stats['matches']['count']}")
    
    elif args.command == "show":
        if args.type == "jobs":
            print("\n📁 Jobs Cache Details")
            print("=" * 50)
            for key, entry in cache_manager.job_cache.items():
                parts = key.split("|")
                print(f"\nQuery: '{parts[0]}' | Location: {parts[1]} | Max: {parts[2]}")
                print(f"  Cached at: {entry.processed_at}")
                print(f"  Job count: {entry.metadata.get('job_count', 0)}")
                if args.verbose and entry.result:
                    print("  Jobs:")
                    for job in entry.result[:3]:
                        print(f"    - {job.get('job_title', 'Unknown')} at {job.get('company', 'Unknown')}")
        
        elif args.type == "resumes":
            print("\n📄 Resumes Cache Details")
            print("=" * 50)
            for key, entry in cache_manager.resume_cache.items():
                file_path = Path(key)
                print(f"\nFile: {file_path.name}")
                print(f"  Path: {key}")
                print(f"  Cached at: {entry.processed_at}")
                print(f"  File size: {entry.metadata.get('size', 0)} bytes")
                if args.verbose and entry.result:
                    result = entry.result.get("processed_resume", {})
                    contact = result.get("contact_info", {})
                    name = contact.get("name", "Unknown")
                    print(f"  Candidate: {name}")
        
        elif args.type == "matches":
            print("\n🔗 Matches Cache Details")
            print("=" * 50)
            for key, entry in cache_manager.match_cache.items():
                print(f"\nCache key: {key[:50]}...")
                print(f"  Cached at: {entry.processed_at}")
                print(f"  Resume count: {entry.metadata.get('resume_count', 0)}")
                print(f"  Job count: {entry.metadata.get('job_count', 0)}")
                print(f"  Match count: {entry.metadata.get('match_count', 0)}")


if __name__ == "__main__":
    main()

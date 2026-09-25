"""
UORCA Dataset Identification CLI Wrapper

This module provides a CLI wrapper for the dataset identification functionality.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

def main():
    """
    Launch UORCA dataset identification functionality.
    """
    # Get path to the main workflow directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # Load environment variables from .env file in project root
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try to find .env file automatically
        load_dotenv(find_dotenv())

    # Check for required environment variables with enhanced user guidance
    missing_vars = []

    if not os.getenv("ENTREZ_EMAIL"):
        missing_vars.append("ENTREZ_EMAIL")

    if not os.getenv("OPENAI_API_KEY"):
        missing_vars.append("OPENAI_API_KEY")

    # Show comprehensive setup instructions if any required variables are missing
    if missing_vars:
        print("\n" + "="*70)
        print("❌ SETUP REQUIRED: Missing essential API credentials for dataset identification")
        print("="*70)

        print("\n📋 What's missing:")
        for var in missing_vars:
            if var == "ENTREZ_EMAIL":
                print("  • ENTREZ_EMAIL - Required by NCBI for accessing GEO/SRA databases")
            elif var == "OPENAI_API_KEY":
                print("  • OPENAI_API_KEY - Required for AI-powered dataset relevance scoring")

        env_file_path = project_root / ".env"
        print(f"\n🔧 How to fix this:")
        print(f"   Create or edit the file: {env_file_path}")

        print(f"\n📝 Add these lines to your .env file:")
        for var in missing_vars:
            if var == "ENTREZ_EMAIL":
                print(f"   {var}=your.email@institution.edu")
            elif var == "OPENAI_API_KEY":
                print(f"   {var}=sk-proj-your-key-here")

        if not os.getenv("ENTREZ_API_KEY"):
            print(f"\n⚡ Optional (for 10x faster processing):")
            print(f"   ENTREZ_API_KEY=your_ncbi_api_key")

        print(f"\n🔗 Where to get API keys:")
        if "OPENAI_API_KEY" in missing_vars:
            print(f"   • OpenAI API key: https://platform.openai.com/api-keys")
            print(f"     (Used for: AI-powered dataset relevance assessment)")
        if not os.getenv("ENTREZ_API_KEY"):
            print(f"   • NCBI API key: https://www.ncbi.nlm.nih.gov/account/settings/")
            print(f"     (Used for: Faster biological database queries - optional but recommended)")

        print(f"\n📋 Complete example .env file:")
        print(f"   # Required for dataset identification")
        print(f"   ENTREZ_EMAIL=researcher@university.edu")
        print(f"   OPENAI_API_KEY=<your-openai-api-key>")
        print(f"   # Optional but recommended")
        print(f"   ENTREZ_API_KEY=your_ncbi_key_here  # For 10x faster processing")

        print("\n💡 Tips:")
        print("   • Keep your .env file secure and don't commit it to version control")
        print("   • Use quotes around values if they contain special characters")
        print("   • The ENTREZ_EMAIL is required by NCBI guidelines for API usage")
        print("   • AI features significantly improve dataset relevance assessment")
        print("="*70)
        sys.exit(1)

    # Show helpful status information when everything is configured
    print("✅ Environment variables configured successfully")
    if not os.getenv("ENTREZ_API_KEY"):
        print("ℹ️  ENTREZ_API_KEY not set - using slower rate limits (3 requests/second)")
        print("   For faster processing, consider getting a free NCBI API key")
    else:
        print("⚡ ENTREZ_API_KEY detected - using faster rate limits")
    print("")

    # Change to project root for proper execution context
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        # Import and run the main function from identification module
        from uorca.identification.dataset_identification import main as identify_main
        identify_main()

    finally:
        # Restore original directory
        os.chdir(original_cwd)

# Re-export the main function
__all__ = ["main"]

if __name__ == "__main__":
    main()

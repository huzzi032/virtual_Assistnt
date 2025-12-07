import sqlite3
import os

try:
    db_path = 'server/database.db'
    if os.path.exists(db_path):
        print(f"✅ Database exists at: {db_path}")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get all table names
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        print('\n📊 Database Tables:')
        for table in tables:
            print(f'  - {table[0]}')
        
        # Check zoom_meeting_sessions table
        table_names = [table[0] for table in tables]
        if 'zoom_meeting_sessions' in table_names:
            print('\n📋 Zoom Meeting Sessions Structure:')
            cur.execute('PRAGMA table_info(zoom_meeting_sessions)')
            columns = cur.fetchall()
            for col in columns:
                print(f'  {col[1]} ({col[2]})')
            
            # Check recent meeting data
            print('\n📄 Recent Zoom Meeting Sessions (last 5):')
            cur.execute('SELECT * FROM zoom_meeting_sessions ORDER BY rowid DESC LIMIT 5')
            meetings = cur.fetchall()
            for i, meeting in enumerate(meetings):
                print(f'  Meeting {i+1}: {meeting}')
        
        # Check zoom_meetings table
        if 'zoom_meetings' in table_names:
            print('\n📋 Zoom Meetings Structure:')
            cur.execute('PRAGMA table_info(zoom_meetings)')
            columns = cur.fetchall()
            for col in columns:
                print(f'  {col[1]} ({col[2]})')
            
            print('\n📄 Recent Zoom Meetings (last 5):')
            cur.execute('SELECT * FROM zoom_meetings ORDER BY rowid DESC LIMIT 5')
            meetings = cur.fetchall()
            for i, meeting in enumerate(meetings):
                print(f'  Meeting {i+1}: {meeting}')
        
        # Check zoom_live_transcripts table
        if 'zoom_live_transcripts' in table_names:
            print('\n📋 Zoom Live Transcripts Structure:')
            cur.execute('PRAGMA table_info(zoom_live_transcripts)')
            columns = cur.fetchall()
            for col in columns:
                print(f'  {col[1]} ({col[2]})')
            
            print('\n📄 Recent Transcripts (last 3):')
            cur.execute('SELECT meeting_id, speaker_name, transcript_content, timestamp FROM zoom_live_transcripts ORDER BY rowid DESC LIMIT 3')
            transcripts = cur.fetchall()
            for i, transcript in enumerate(transcripts):
                print(f'  Transcript {i+1}: Meeting {transcript[0]} - {transcript[1]}: {transcript[2][:100]}... at {transcript[3]}')
        
        conn.close()
    else:
        print(f"❌ Database not found at: {db_path}")
        
except Exception as e:
    import traceback
    print(f'❌ Database Error: {e}')
    print(f'Traceback: {traceback.format_exc()}')
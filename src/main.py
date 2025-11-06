"""
Leetcode Investment Tracker - Backend API with Flask
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime, timedelta
from typing import Optional, List
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import random
import json
import leetcode
import leetcode.auth
from dotenv import load_dotenv
import os

load_dotenv()
Base = declarative_base()
POSTGRES_URL = os.getenv('POSTGRES_URL')

# Configuration
DIFFICULTY_VALUES = {
    'Easy': 10.0,
    'Medium': 25.0,
    'Hard': 50.0
}
DIFFICULTY_TOKENS = {
    'Easy': 1,
    'Medium': 1,
    'Hard': 1
}
# Prize ranges for each token type
PRIZE_RANGES = {
    'Easy': {
        'cash': (1, 20),
        'multiplier': (0.01, 0.2)
    },
    'Medium': {
        'cash': (5, 40),
        'multiplier': (0.05, 0.4)
    },
    'Hard': {
        'cash': (15, 100),
        'multiplier': (0.15, 1.0)
    }
}
DAILY_DECAY_RATE = 0.02  # 2% decay per day without solving
STREAK_MULTIPLIER = 0.04  # 4% bonus per day streak


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    leetcode_username = Column(String)
    leetcode_session = Column(String)
    total_investment = Column(Float, default=0.0)
    total_multiplier = Column(Float, default=1.0)  # Accumulated multiplier from spins
    tokens_easy = Column(Integer, default=0)
    tokens_medium = Column(Integer, default=0)
    tokens_hard = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    max_streak = Column(Integer, default=0)
    last_solved_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    problems = relationship('SolvedProblem', back_populates='user', cascade='all, delete-orphan')
    investment_history = relationship('InvestmentHistory', back_populates='user', cascade='all, delete-orphan')
    spins = relationship('WheelSpin', back_populates='user', cascade='all, delete-orphan')


class SolvedProblem(Base):
    __tablename__ = 'solved_problems'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    problem_id = Column(String, nullable=False)
    problem_slug = Column(String)  # Slug for LeetCode URL (e.g., "two-sum")
    title = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    investment_value = Column(Float, nullable=False)
    tokens_earned = Column(Integer, nullable=False)
    topics = Column(Text)  # Store topics as JSON string or comma-separated
    solved_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='problems')


class InvestmentHistory(Base):
    __tablename__ = 'investment_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    investment_amount = Column(Float, nullable=False)
    action = Column(String, nullable=False)
    description = Column(String)
    
    user = relationship('User', back_populates='investment_history')


class WheelSpin(Base):
    __tablename__ = 'wheel_spins'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    token_type = Column(String, nullable=False)  # 'Easy', 'Medium', or 'Hard'
    spin_type = Column(String, nullable=False)  # 'cash' or 'multiplier'
    amount = Column(Float, nullable=False)  # Cash amount or multiplier percentage
    spun_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='spins')


class LeetcodeService:
    """Service for interacting with Leetcode API"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id
        self.api_instance = None
        
        if session_id:
            csrf_token = leetcode.auth.get_csrf_cookie(session_id)
            configuration = leetcode.Configuration()
            configuration.api_key["x-csrftoken"] = csrf_token
            configuration.api_key["csrftoken"] = csrf_token
            configuration.api_key["LEETCODE_SESSION"] = session_id
            configuration.api_key["Referer"] = "https://leetcode.com"
            configuration.debug = False
            self.api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))
    
    def get_problem_info(self, title_slug: str) -> Optional[dict]:
        """Fetch problem information using public API (no auth needed)"""
        try:
            url = "https://leetcode.com/graphql"
            query = """
            query getQuestionDetail($titleSlug: String!) {
                question(titleSlug: $titleSlug) {
                    questionId
                    title
                    difficulty
                    topicTags {
                        name
                        slug
                    }
                }
            }
            """
            
            response = requests.post(
                url,
                json={'query': query, 'variables': {'titleSlug': title_slug}},
                headers={
                    'Content-Type': 'application/json',
                    'Referer': 'https://leetcode.com'
                },
                timeout=10
            )
            
            data = response.json()
            if data.get('data') and data['data'].get('question'):
                q = data['data']['question']
                topics = [tag['name'] for tag in q.get('topicTags', [])]
                return {
                    'questionId': q['questionId'],
                    'title': q['title'],
                    'difficulty': q['difficulty'],
                    'topics': topics
                }
            
            print(f"API Response: {data}")
            return None
        except Exception as e:
            print(f"Error fetching problem from Leetcode: {e}")
            import traceback
            traceback.print_exc()
            return None


class LeetcodeTracker:
    def __init__(self, database_url: str = POSTGRES_URL):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self._migrate_database()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def _migrate_database(self):
        """Add missing columns to existing tables"""
        try:
            with self.engine.begin() as conn:  # begin() automatically commits or rolls back
                # Check if problem_slug column exists, if not add it
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='solved_problems' AND column_name='problem_slug'
                """))
                if result.fetchone() is None:
                    print("Adding problem_slug column to solved_problems table...")
                    conn.execute(text("ALTER TABLE solved_problems ADD COLUMN problem_slug VARCHAR"))
                    print("Migration completed successfully.")
        except Exception as e:
            # If migration fails, it's okay - column might already exist or table might not exist yet
            print(f"Migration check completed (column may already exist): {e}")
    
    def create_user(self, username: str, leetcode_username: str = None, 
                   leetcode_session: str = None) -> User:
        user = User(
            username=username,
            leetcode_username=leetcode_username or username,
            leetcode_session=leetcode_session
        )
        self.session.add(user)
        self.session.commit()
        return user
    
    def get_user(self, username: str) -> Optional[User]:
        try:
            return self.session.query(User).filter_by(username=username).first()
        except Exception as e:
            print(f"Error getting user {username}: {e}")
            self.session.rollback()
            # Try once more after rollback
            try:
                return self.session.query(User).filter_by(username=username).first()
            except Exception as e2:
                print(f"Error after rollback: {e2}")
                self.session.rollback()
                return None
    
    def get_all_users(self) -> List[User]:
        return self.session.query(User).all()
    
    def delete_user(self, username: str) -> bool:
        """Delete a user and all their data"""
        user = self.get_user(username)
        if user:
            self.session.delete(user)
            self.session.commit()
            print(f"Deleted user: {username}")
            return True
        return False
    
    def apply_daily_decay(self, user: User):
        if not user.last_solved_date:
            return
        
        try:
            # Check the most recent decay entry to see when decay was last applied
            last_decay = self.session.query(InvestmentHistory)\
                .filter_by(user_id=user.id, action='decay')\
                .order_by(InvestmentHistory.date.desc())\
                .first()
        except Exception as e:
            print(f"Error checking decay history: {e}")
            self.session.rollback()
            return
        
        # Determine the reference date: last decay date or last solved date
        if last_decay:
            # Use the date of the last decay (only date part, ignore time)
            last_decay_date = last_decay.date.date()
            today = datetime.utcnow().date()
            
            # If decay was already applied today, don't apply again
            if last_decay_date >= today:
                return
            
            # Calculate days since last decay
            days_since_last_decay = (today - last_decay_date).days
            days_to_apply = days_since_last_decay
        else:
            # No previous decay, calculate from last solved date
            days_inactive = (datetime.utcnow().date() - user.last_solved_date.date()).days
            days_to_apply = days_inactive
        
        # Only apply decay if there are days to decay and user has investment
        if days_to_apply > 0 and user.total_investment > 0:
            for _ in range(days_to_apply):
                decay_amount = user.total_investment * DAILY_DECAY_RATE
                user.total_investment = max(0, user.total_investment - decay_amount)
                
                history = InvestmentHistory(
                    user_id=user.id,
                    investment_amount=-decay_amount,
                    action='decay',
                    description=f'Daily decay ({DAILY_DECAY_RATE*100}%)'
                )
                self.session.add(history)
            
            user.current_streak = 0
            self.session.commit()
    
    def calculate_streak_bonus(self, base_value: float, streak: int) -> float:
        return base_value * (1 + (streak * STREAK_MULTIPLIER))
    
    def spin_wheel(self, username: str, token_type: str) -> Optional[dict]:
        """Spin the wheel using one token of specified type"""
        user = self.get_user(username)
        if not user:
            return None
        
        # Validate token type
        if token_type not in ['Easy', 'Medium', 'Hard']:
            return None
        
        # Check if user has token of this type
        token_attr = f'tokens_{token_type.lower()}'
        if getattr(user, token_attr) < 1:
            return None
        
        # Deduct token
        setattr(user, token_attr, getattr(user, token_attr) - 1)
        
        # 80% chance for cash, 20% for multiplier
        spin_type = 'cash' if random.random() < 0.8 else 'multiplier'
        
        # Get prize range for this token type
        prize_range = PRIZE_RANGES[token_type]
        
        if spin_type == 'cash':
            # Cash prize based on token type
            cash_min, cash_max = prize_range['cash']
            amount = round(random.uniform(cash_min, cash_max), 2)
            user.total_investment += amount
            
            # Record in history
            history = InvestmentHistory(
                user_id=user.id,
                investment_amount=amount,
                action='wheel_spin',
                description=f'Wheel spin ({token_type} token): Won ${amount} cash!'
            )
            self.session.add(history)
            
        else:  # multiplier
            # Multiplier based on token type
            mult_min, mult_max = prize_range['multiplier']
            multiplier_percent = round(random.uniform(mult_min, mult_max), 2)
            multiplier_decimal = multiplier_percent / 100
            
            # Add to total multiplier
            user.total_multiplier += multiplier_decimal
            
            # Apply to current investment
            bonus = user.total_investment * multiplier_decimal
            user.total_investment += bonus
            
            # Record in history
            history = InvestmentHistory(
                user_id=user.id,
                investment_amount=bonus,
                action='wheel_spin',
                description=f'Wheel spin ({token_type} token): Won +{multiplier_percent}% multiplier! (Applied: ${bonus:.2f})'
            )
            self.session.add(history)
            amount = multiplier_percent
        
        # Record spin
        spin = WheelSpin(
            user_id=user.id,
            token_type=token_type,
            spin_type=spin_type,
            amount=amount
        )
        self.session.add(spin)
        self.session.commit()
        
        return {
            'spin_type': spin_type,
            'amount': amount,
            'token_type': token_type,
            'tokens_remaining': {
                'Easy': user.tokens_easy,
                'Medium': user.tokens_medium,
                'Hard': user.tokens_hard
            },
            'total_investment': round(user.total_investment, 2),
            'total_multiplier': round(user.total_multiplier * 100, 2)  # As percentage
        }
    
    def add_solved_problem(self, username: str, problem_title_slug: str, 
                          difficulty: str = None) -> Optional[dict]:
        user = self.get_user(username)
        if not user:
            print(f"User not found: {username}")
            return None
        
        self.apply_daily_decay(user)
        
        # Fetch problem info
        topics = []
        if not difficulty:
            service = LeetcodeService(user.leetcode_session)
            problem_info = service.get_problem_info(problem_title_slug)
            if problem_info:
                difficulty = problem_info['difficulty']
                title = problem_info['title']
                problem_id = problem_info['questionId']
                topics = problem_info.get('topics', [])
                print(f"Fetched problem: {title} ({difficulty}) - Topics: {topics}")
            else:
                print(f"Could not fetch problem info for: {problem_title_slug}")
                print("Attempting with manual difficulty...")
                difficulty = 'Medium'
                title = problem_title_slug.replace('-', ' ').title()
                problem_id = problem_title_slug
        else:
            # Still try to fetch topics even if difficulty is provided
            service = LeetcodeService(user.leetcode_session)
            problem_info = service.get_problem_info(problem_title_slug)
            if problem_info:
                topics = problem_info.get('topics', [])
            title = problem_title_slug.replace('-', ' ').title()
            problem_id = problem_title_slug
        
        # Validate difficulty
        if difficulty not in DIFFICULTY_VALUES:
            print(f"Invalid difficulty: {difficulty}")
            return None
        
        # Check if solved today
        today = datetime.utcnow().date()
        last_solved_date = user.last_solved_date.date() if user.last_solved_date else None
        
        # Update streak
        if last_solved_date == today:
            pass
        elif last_solved_date == today - timedelta(days=1):
            user.current_streak += 1
        else:
            user.current_streak = 1
        
        # Calculate investment with multiplier
        base_value = DIFFICULTY_VALUES.get(difficulty, 10.0)
        investment_value = self.calculate_streak_bonus(base_value, user.current_streak)
        investment_value *= user.total_multiplier  # Apply accumulated multiplier
        
        # Award tokens (1 token of the appropriate difficulty)
        tokens_earned = DIFFICULTY_TOKENS.get(difficulty, 1)
        token_attr = f'tokens_{difficulty.lower()}'
        setattr(user, token_attr, getattr(user, token_attr) + tokens_earned)
        
        # Create records
        # Store topics as JSON string
        topics_json = json.dumps(topics) if topics else None
        
        problem = SolvedProblem(
            user_id=user.id,
            problem_id=problem_id,
            problem_slug=problem_title_slug,  # Store the slug for URL construction
            title=title,
            difficulty=difficulty,
            investment_value=investment_value,
            tokens_earned=tokens_earned,
            topics=topics_json
        )
        self.session.add(problem)
        
        user.total_investment += investment_value
        user.last_solved_date = datetime.utcnow()
        user.max_streak = max(user.max_streak, user.current_streak)
        
        history = InvestmentHistory(
            user_id=user.id,
            investment_amount=investment_value,
            action='solve',
            description=f'Solved {difficulty} problem: {title} (+{tokens_earned} {difficulty} token{"s" if tokens_earned > 1 else ""})'
        )
        self.session.add(history)
        
        try:
            self.session.commit()
            print(f"Successfully added problem: {title}")
        except Exception as e:
            print(f"Error committing to database: {e}")
            self.session.rollback()
            return None
        
        return {
            'problem': {
                'id': problem.id,
                'title': title,
                'difficulty': difficulty,
                'investment_value': round(investment_value, 2),
                'tokens_earned': tokens_earned
            },
            'user_stats': self.get_user_stats(username)
        }
    
    def get_user_stats(self, username: str) -> Optional[dict]:
        user = self.get_user(username)
        if not user:
            return None
        
        self.apply_daily_decay(user)
        
        problems_by_difficulty = {
            'Easy': len([p for p in user.problems if p.difficulty == 'Easy']),
            'Medium': len([p for p in user.problems if p.difficulty == 'Medium']),
            'Hard': len([p for p in user.problems if p.difficulty == 'Hard'])
        }
        
        return {
            'username': user.username,
            'total_investment': round(user.total_investment, 2),
            'total_multiplier': round(user.total_multiplier * 100, 2),  # As percentage
            'tokens': {
                'Easy': user.tokens_easy,
                'Medium': user.tokens_medium,
                'Hard': user.tokens_hard
            },
            'current_streak': user.current_streak,
            'max_streak': user.max_streak,
            'total_problems': len(user.problems),
            'problems_by_difficulty': problems_by_difficulty,
            'last_solved': user.last_solved_date.isoformat() if user.last_solved_date else None
        }
    
    def get_recent_problems(self, username: str, limit: int = 10, topic_filter: str = None) -> List[dict]:
        user = self.get_user(username)
        if not user:
            return []
        
        try:
            query = self.session.query(SolvedProblem)\
                .filter_by(user_id=user.id)
            
            # Apply topic filter if provided
            if topic_filter:
                # Filter problems that have the topic in their topics JSON
                query = query.filter(
                    SolvedProblem.topics.ilike(f'%{topic_filter}%')
                )
            
            problems = query.order_by(SolvedProblem.solved_at.desc())\
                .limit(limit)\
                .all()
        except Exception as e:
            print(f"Error getting recent problems: {e}")
            self.session.rollback()
            return []
        
        result = []
        for p in problems:
            topics = []
            if p.topics:
                try:
                    topics = json.loads(p.topics)
                except:
                    # Fallback if topics is not JSON (for old data)
                    topics = [p.topics] if p.topics else []
            
            # Construct LeetCode URL from slug, fallback to problem_id if slug not available
            leetcode_url = None
            if p.problem_slug:
                leetcode_url = f"https://leetcode.com/problems/{p.problem_slug}/"
            elif p.problem_id and not p.problem_id.isdigit():
                # If problem_id looks like a slug (not numeric), use it
                leetcode_url = f"https://leetcode.com/problems/{p.problem_id}/"
            
            result.append({
                'id': p.id,
                'problem_id': p.problem_id,
                'title': p.title,
                'difficulty': p.difficulty,
                'investment_value': round(p.investment_value, 2),
                'tokens_earned': p.tokens_earned,
                'topics': topics,
                'solved_at': p.solved_at.isoformat(),
                'leetcode_url': leetcode_url
            })
        
        return result
    
    def get_available_topics(self, username: str) -> List[str]:
        """Get all unique topics from a user's solved problems"""
        user = self.get_user(username)
        if not user:
            return []
        
        try:
            problems = self.session.query(SolvedProblem)\
                .filter_by(user_id=user.id)\
                .all()
            
            topics_set = set()
            for p in problems:
                if p.topics:
                    try:
                        topics = json.loads(p.topics)
                        topics_set.update(topics)
                    except:
                        # Fallback if topics is not JSON (for old data)
                        if p.topics:
                            topics_set.add(p.topics)
            
            return sorted(list(topics_set))
        except Exception as e:
            print(f"Error getting available topics: {e}")
            self.session.rollback()
            return []
    
    def get_investment_history(self, username: str, limit: int = 20) -> List[dict]:
        user = self.get_user(username)
        if not user:
            return []
        
        try:
            history = self.session.query(InvestmentHistory)\
                .filter_by(user_id=user.id)\
                .order_by(InvestmentHistory.date.desc())\
                .limit(limit)\
                .all()
            
            return [{
                'id': h.id,
                'date': h.date.isoformat(),
                'amount': round(h.investment_amount, 2),
                'action': h.action,
                'description': h.description
            } for h in history]
        except Exception as e:
            print(f"Error getting investment history: {e}")
            self.session.rollback()
            return []
    
    def get_spin_history(self, username: str, limit: int = 10) -> List[dict]:
        user = self.get_user(username)
        if not user:
            return []
        
        spins = self.session.query(WheelSpin)\
            .filter_by(user_id=user.id)\
            .order_by(WheelSpin.spun_at.desc())\
            .limit(limit)\
            .all()
        
        return [{
            'id': s.id,
            'token_type': s.token_type,
            'spin_type': s.spin_type,
            'amount': round(s.amount, 2),
            'spun_at': s.spun_at.isoformat()
        } for s in spins]
    
    def delete_problem(self, username: str, problem_id: int) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        
        problem = self.session.query(SolvedProblem)\
            .filter_by(id=problem_id, user_id=user.id)\
            .first()
        
        if problem:
            user.total_investment = max(0, user.total_investment - problem.investment_value)
            
            # Deduct the appropriate token type
            token_attr = f'tokens_{problem.difficulty.lower()}'
            setattr(user, token_attr, max(0, getattr(user, token_attr) - problem.tokens_earned))
            
            history = InvestmentHistory(
                user_id=user.id,
                investment_amount=-problem.investment_value,
                action='delete',
                description=f'Removed problem: {problem.title}'
            )
            self.session.add(history)
            
            self.session.delete(problem)
            self.session.commit()
            return True
        
        return False
    
    def get_top_earners(self, limit: int = 10) -> List[dict]:
        """Get top users by total_investment"""
        users = self.session.query(User)\
            .order_by(User.total_investment.desc())\
            .limit(limit)\
            .all()
        
        return [{
            'username': user.username,
            'total_investment': round(user.total_investment, 2),
            'total_problems': len(user.problems),
            'current_streak': user.current_streak,
            'max_streak': user.max_streak
        } for user in users]


# Flask API
app = Flask(__name__)
CORS(app)

tracker = LeetcodeTracker()

@app.route('/')
def index():
    return jsonify({
        'message': 'Leetcode Investment Tracker API',
        'endpoints': {
            'users': '/api/users',
            'user_stats': '/api/users/<username>/stats',
            'problems': '/api/users/<username>/problems',
            'history': '/api/users/<username>/history',
            'spin': '/api/users/<username>/spin',
            'leaderboard': '/api/leaderboard'
        }
    })

@app.route('/api/users', methods=['GET', 'POST'])
def users():
    if request.method == 'POST':
        data = request.json
        user = tracker.create_user(
            data['username'],
            data.get('leetcode_username'),
            data.get('leetcode_session')
        )
        return jsonify({'username': user.username}), 201
    else:
        users = tracker.get_all_users()
        return jsonify([{'username': u.username} for u in users])

@app.route('/api/users/<username>', methods=['DELETE'])
def delete_user(username):
    if tracker.delete_user(username):
        return jsonify({'success': True, 'message': f'User {username} deleted'})
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/users/<username>/stats', methods=['GET'])
def user_stats(username):
    stats = tracker.get_user_stats(username)
    if stats:
        return jsonify(stats)
    return jsonify({'error': 'User not found'}), 404

@app.route('/api/users/<username>/problems', methods=['GET', 'POST'])
def problems(username):
    if request.method == 'POST':
        data = request.json
        print(f"Received problem data: {data}")
        
        if not data.get('problem_slug'):
            return jsonify({'error': 'problem_slug is required'}), 400
        
        result = tracker.add_solved_problem(
            username,
            data['problem_slug'],
            data.get('difficulty') if data.get('difficulty') else None
        )
        
        if result:
            return jsonify(result), 201
        return jsonify({'error': 'Failed to add problem. Check server logs for details.'}), 400
    else:
        limit = request.args.get('limit', 10, type=int)
        topic_filter = request.args.get('topic', None, type=str)
        problems = tracker.get_recent_problems(username, limit, topic_filter)
        return jsonify(problems)

@app.route('/api/users/<username>/problems/<int:problem_id>', methods=['DELETE'])
def delete_problem(username, problem_id):
    if tracker.delete_problem(username, problem_id):
        return jsonify({'success': True})
    return jsonify({'error': 'Problem not found'}), 404

@app.route('/api/users/<username>/topics', methods=['GET'])
def get_topics(username):
    topics = tracker.get_available_topics(username)
    return jsonify(topics)

@app.route('/api/users/<username>/history', methods=['GET'])
def history(username):
    limit = request.args.get('limit', 20, type=int)
    history = tracker.get_investment_history(username, limit)
    return jsonify(history)

@app.route('/api/users/<username>/spin', methods=['POST'])
def spin_wheel(username):
    data = request.json or {}
    token_type = data.get('token_type', 'Easy')
    
    result = tracker.spin_wheel(username, token_type)
    if result:
        return jsonify(result), 200
    return jsonify({'error': 'Not enough tokens of that type or user not found'}), 400

@app.route('/api/users/<username>/spins', methods=['GET'])
def spin_history(username):
    limit = request.args.get('limit', 10, type=int)
    spins = tracker.get_spin_history(username, limit)
    return jsonify(spins)

@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    limit = request.args.get('limit', 10, type=int)
    top_earners = tracker.get_top_earners(limit)
    return jsonify(top_earners)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
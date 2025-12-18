# Customer-Service-Chatbot
"""
CUSTOMER SERVICE CHATBOT - Complete Implementation
Rule-based NLP Chatbot for Customer Service Automation
Author: Your Name
Date: December 18, 2025
"""

import re
from collections import Counter

class CustomerServiceChatbot:
    """
    A rule-based chatbot using NLP techniques for customer service automation.
    Implements pattern matching, intent recognition, and confidence scoring.
    """
    
    def __init__(self):
        """Initialize the chatbot with intents, patterns, and responses."""
        
        # Define intents with patterns and responses
        self.intents = {
            "greeting": {
                "patterns": [
                    r"(hello|hi|hey|greetings|good morning|good afternoon|good evening)",
                    r"(what's up|howdy|hola)",
                    r"(nice to meet|glad to talk)"
                ],
                "responses": [
                    "Hello! 👋 Welcome to our customer service. How can I help you today?",
                    "Hi there! 😊 What can I assist you with?",
                    "Greetings! I'm here to help. What do you need?",
                    "Welcome! Please let me know how I can serve you."
                ]
            },
            
            "product_info": {
                "patterns": [
                    r"(tell|show|what).*(product|item|product line)",
                    r"(product information|product details|what do you sell)",
                    r"(available products|do you have|product catalog)",
                    r"(price|cost|how much).*(product|item)"
                ],
                "responses": [
                    "We offer a variety of products! 📦 Could you be more specific? Are you looking for electronics, clothing, or something else?",
                    "Our product range includes electronics, home appliances, fashion, and more! What interests you?",
                    "I'd be happy to help! We have multiple categories. What type of product are you interested in?",
                    "We have many products available. Please specify the category you're interested in!"
                ]
            },
            
            "order_status": {
                "patterns": [
                    r"(where.*order|order status|track|tracking)",
                    r"(when.*arrive|delivery|shipped)",
                    r"(order.*arrived|order.*coming)",
                    r"(my order|check order)"
                ],
                "responses": [
                    "To check your order status, I'll need your order number. Could you provide it? 📍",
                    "I can help track your order! Please share your order ID.",
                    "I'd be happy to help! What's your order number?",
                    "To track your package, I need your order number. Can you provide it?"
                ]
            },
            
            "returns_refunds": {
                "patterns": [
                    r"(return|refund|money back)",
                    r"(not satisfied|don't like|broken|damaged)",
                    r"(return policy|how to return)",
                    r"(exchange|replacement)"
                ],
                "responses": [
                    "I understand! 🔄 We accept returns within 30 days. Would you like to initiate a return?",
                    "We have a hassle-free return policy. Can you provide your order number?",
                    "I'm sorry you're unsatisfied. Let me help you with a return or refund process.",
                    "Returns are easy! Please provide your order ID and reason for return."
                ]
            },
            
            "billing": {
                "patterns": [
                    r"(billing|invoice|payment|charge)",
                    r"(why was.*charged|duplicate charge|wrong amount)",
                    r"(payment method|update payment)",
                    r"(card declined|payment failed)"
                ],
                "responses": [
                    "I can help with billing questions! 💳 What's the issue?",
                    "For billing inquiries, I can assist. Please provide your order number.",
                    "Let me help resolve your billing concern. Can you share more details?",
                    "I'm here to help with payment issues. What happened?"
                ]
            },
            
            "support_hours": {
                "patterns": [
                    r"(hours|open|closed|available)",
                    r"(support hours|customer service hours)",
                    r"(when.*available|when.*open)",
                    r"(contact support|reach support)"
                ],
                "responses": [
                    "⏰ Our customer support is available 24/7! You can always reach us.",
                    "We're open round the clock for your convenience!",
                    "Our support team is available 24/7 to assist you.",
                    "We're here to help anytime, day or night!"
                ]
            },
            
            "goodbye": {
                "patterns": [
                    r"(goodbye|bye|see you|farewell)",
                    r"(exit|quit|close|end)",
                    r"(that's all|nothing else|thank you|thanks)",
                    r"(have a nice|good.*day)"
                ],
                "responses": [
                    "Thank you for contacting us! Goodbye! 👋",
                    "It was great helping you. Have a wonderful day!",
                    "Bye! We appreciate your business.",
                    "Take care! Feel free to reach out anytime. 😊"
                ]
            }
        }
    
    def preprocess_text(self, text):
        """
        Preprocess user input: lowercase, remove punctuation, tokenize.
        
        Args:
            text (str): Raw user input
            
        Returns:
            tuple: (cleaned_text, tokens)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize (split into words)
        tokens = text.split()
        
        return text, tokens
    
    def calculate_similarity(self, user_tokens, pattern_tokens):
        """
        Calculate Jaccard similarity between two token sets.
        Jaccard Similarity = |Intersection| / |Union|
        
        Args:
            user_tokens (list): Tokens from user input
            pattern_tokens (list): Tokens from pattern
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        user_set = set(user_tokens)
        pattern_set = set(pattern_tokens)
        
        if len(user_set.union(pattern_set)) == 0:
            return 0.0
        
        intersection = len(user_set.intersection(pattern_set))
        union = len(user_set.union(pattern_set))
        
        return intersection / union
    
    def extract_intent(self, user_input):
        """
        Extract intent from user input using pattern matching and similarity scoring.
        
        Args:
            user_input (str): User's message
            
        Returns:
            tuple: (intent_name, confidence_level, confidence_score)
        """
        cleaned_text, user_tokens = self.preprocess_text(user_input)
        
        best_intent = None
        best_score = 0.0
        regex_match = False
        
        # Check each intent
        for intent_name, intent_data in self.intents.items():
            patterns = intent_data["patterns"]
            
            # Method 1: Regex pattern matching (highest priority)
            for pattern in patterns:
                if re.search(pattern, cleaned_text):
                    best_intent = intent_name
                    best_score = 1.0  # Perfect match
                    regex_match = True
                    break
            
            if regex_match:
                break
            
            # Method 2: Token similarity matching (fallback)
            for pattern in patterns:
                # Extract tokens from regex pattern
                pattern_text = re.sub(r'[^\w\s|()]', '', pattern)
                pattern_options = pattern_text.split('|')
                
                for option in pattern_options:
                    option = option.strip()
                    if not option or option in ['', ' ']:
                        continue
                    
                    pattern_tokens = option.split()
                    similarity = self.calculate_similarity(user_tokens, pattern_tokens)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = intent_name
        
        # Determine confidence level
        if best_score >= 0.8:
            confidence = "High"
        elif best_score >= 0.5:
            confidence = "Medium"
        elif best_score > 0.0:
            confidence = "Low"
        else:
            confidence = "No Match"
        
        return best_intent, confidence, best_score
    
    def get_response(self, intent_name):
        """
        Get a response for the identified intent.
        
        Args:
            intent_name (str): Name of the recognized intent
            
        Returns:
            str: Response message
        """
        if intent_name and intent_name in self.intents:
            import random
            responses = self.intents[intent_name]["responses"]
            return random.choice(responses)
        
        return "I'm not sure I understood that. Could you rephrase? 🤔 Or type 'help' for assistance."
    
    def chat(self, user_input):
        """
        Process user input and generate response.
        
        Args:
            user_input (str): User's message
            
        Returns:
            dict: Contains response, intent, and confidence
        """
        intent, confidence, score = self.extract_intent(user_input)
        response = self.get_response(intent)
        
        return {
            "user_input": user_input,
            "intent": intent if intent else "unknown",
            "confidence": confidence,
            "confidence_score": f"{score:.2f}",
            "response": response
        }
    
    def display_result(self, result):
        """Display chat result in formatted way."""
        print("\n" + "="*60)
        print(f"👤 User: {result['user_input']}")
        print(f"🤖 Bot: {result['response']}")
        print(f"🎯 Intent: {result['intent']} | Confidence: {result['confidence']} ({result['confidence_score']})")
        print("="*60)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main function to run the chatbot."""
    print("\n" + "="*60)
    print("🤖 CUSTOMER SERVICE CHATBOT - NLP POWERED")
    print("="*60)
    print("Welcome! I'm here to help with your customer service needs.")
    print("Type 'quit' or 'bye' to exit. Type 'help' for commands.\n")
    print("="*60 + "\n")
    
    # Initialize chatbot
    chatbot = CustomerServiceChatbot()
    
    # Example queries for demonstration
    example_queries = [
        "Hello! I need some help",
        "What products do you have?",
        "Where is my order? Order number is #12345",
        "I want to return this broken item",
        "I was charged twice. This is a duplicate charge!",
        "What are your support hours?",
        "Thank you so much! Goodbye!"
    ]
    
    print("📝 DEMONSTRATION MODE - Running Example Queries:\n")
    
    for query in example_queries:
        result = chatbot.chat(query)
        chatbot.display_result(result)
        input("\nPress Enter to continue...")
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60 + "\n")
    
    # Interactive mode
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Bot: Thank you for using our service. Goodbye! 👋\n")
                break
            
            if user_input.lower() == 'help':
                print("\n📋 AVAILABLE COMMANDS:")
                print("- Type any customer service question")
                print("- 'quit' or 'bye' to exit")
                print("- 'help' to see this menu\n")
                continue
            
            result = chatbot.chat(user_input)
            chatbot.display_result(result)
        
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Thank you! Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")


# ============================================================================
# TECHNICAL DOCUMENTATION
# ============================================================================

"""
NLP TECHNIQUES IMPLEMENTED:

1. TEXT PREPROCESSING
   - Lowercase conversion: Normalizes text to lowercase
   - Punctuation removal: Removes special characters
   - Tokenization: Splits text into individual words

2. PATTERN MATCHING
   - Regular expressions (regex) for exact phrase detection
   - Priority-based matching for accuracy
   - Multiple patterns per intent

3. SIMILARITY SCORING
   - Jaccard Similarity Coefficient: Measures overlap between token sets
   - Formula: |Intersection| / |Union|
   - Provides confidence levels (High/Medium/Low)

4. INTENT RECOGNITION
   - Two-tier matching system:
     a) Regex patterns (highest priority, exact matches)
     b) Token similarity (fallback, approximate matches)
   - Confidence scoring provides transparency

5. RESPONSE GENERATION
   - Intent-based response selection
   - Multiple response variations for naturalness
   - Fallback responses for unmatched queries

ARCHITECTURE:
┌─────────────────────┐
│   User Input        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Preprocessing       │
│ (lowercase, remove  │
│  punctuation,       │
│  tokenize)          │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Intent Extraction   │
│ (regex + similarity)│
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Confidence Scoring  │
│ (High/Med/Low)      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Response Selection  │
│ (intent-based)      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Bot Response        │
└─────────────────────┘

INTENT CATEGORIES:
1. greeting - Initial conversation and welcomes
2. product_info - Product inquiries and catalog
3. order_status - Order tracking and delivery
4. returns_refunds - Returns and refund processes
5. billing - Payment and billing issues
6. support_hours - Operational hours and contact
7. goodbye - Conversation closure

USE CASES:
- Customer support automation
- FAQ handling
- Order status inquiries
- Basic troubleshooting
- Lead qualification

SCALABILITY PATHS:
- Machine Learning: Use NLTK/spaCy for NER and sentiment
- Deep Learning: Implement transformers (BERT, GPT)
- Database Integration: Store resolved issues and patterns
- API Integration: Connect to order management systems
- Multi-language Support: Add language detection and translation
"""


if __name__ == "__main__":
    main()

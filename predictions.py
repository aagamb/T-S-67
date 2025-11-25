from policy_proposal_labeler import predict_post, predict_batch, CustomEnsemble
import pandas as pd
import sys

sys.modules['__main__'].CustomEnsemble = CustomEnsemble


clear_scams = [
    {
        "id": "scam_1",
        "text": "🎉 CONGRATULATIONS! You've been selected as a Bitcoin lottery winner! You've won 50 BTC! "
                "To claim your prize, send 0.5 BTC for processing fees to our wallet: 9X8Y7Z6W5V4U3T2S1R. "
                "Wire transfer your payment via Western Union today!",
        "description": "Lottery scam with upfront fee requirement"
    },
    {
        "id": "scam_2",
        "text": "💰 EXCLUSIVE INVESTMENT OPPORTUNITY 💰 Join our Ethereum trading pool! Minimum investment $500, "
                "guaranteed 300% returns monthly! Risk-free! Already 10,000 members making profits! "
                "Send money to our PayPal: cryptomaster@profits.com. LIMITED SPOTS AVAILABLE!",
        "description": "Ponzi scheme with unrealistic returns and urgency"
    },
]


clear_non_scams = [
    {
        "id": "legit_1",
        "text": "Just finished reading the Bitcoin whitepaper again. Satoshi's vision of a peer-to-peer electronic "
                "cash system is still revolutionary. The mathematical elegance of proof-of-work consensus "
                "is impressive. What are your thoughts on the original design?",
        "description": "Educational discussion about cryptocurrency technology"
    },
    {
        "id": "legit_2",
        "text": "Ethereum's transition to proof-of-stake has reduced energy consumption by over 99%. "
                "This is a significant milestone for blockchain sustainability. The merge was completed "
                "successfully without major issues. Looking forward to future upgrades like sharding.",
        "description": "Informative post about legitimate blockchain developments"
    }
]


ambiguous_cases = [
    {
        "id": "ambiguous_1",
        "text": "Lost access to my old wallet from 2016. Anyone know good recovery services? "
                "Willing to pay 20% of recovered funds as fee. Had significant amount of BTC. "
                "Need someone trustworthy who can help. Please DM if you have experience with this.",
        "description": "Could be genuine help request or setup for recovery scam"
    },
    {
        "id": "ambiguous_2",
        "text": "Urgent: Bitcoin is crashing! Sell now before you lose everything! "
                "Market manipulation by whales! This is not a drill! Protect your investments! "
                "Switch to stable coins immediately!",
        "description": "Panic-inducing but could be genuine concern or FUD spreading"
    }
]


def run_predictions():
    print("=" * 80)
    print("CRYPTO SCAM DETECTION - TEST PREDICTIONS")
    print("=" * 80)
    print()
    
    all_results = []
    
    print("📛 TESTING CLEAR SCAMS")
    print("-" * 80)
    for item in clear_scams:
        prediction, probability = predict_post(item["text"], return_probability=True)
        label = "🚨 SCAM" if prediction == 1 else "✅ LEGIT"
        confidence = probability if prediction == 1 else (1 - probability)
        
        print(f"\n{item['id']}: {label} (Confidence: {confidence:.1%})")
        print(f"Description: {item['description']}")
        print(f"Text: {item['text'][:150]}...")
        
        all_results.append({
            "id": item["id"],
            "category": "Clear Scam",
            "text": item["text"],
            "description": item["description"],
            "prediction": "Scam" if prediction == 1 else "Legit",
            "probability": probability,
            "confidence": confidence
        })
    
    print("\n\n")
    print("✅ TESTING CLEAR NON-SCAMS")
    print("-" * 80)
    for item in clear_non_scams:
        prediction, probability = predict_post(item["text"], return_probability=True)
        label = "🚨 SCAM" if prediction == 1 else "✅ LEGIT"
        confidence = probability if prediction == 1 else (1 - probability)
        
        print(f"\n{item['id']}: {label} (Confidence: {confidence:.1%})")
        print(f"Description: {item['description']}")
        print(f"Text: {item['text'][:150]}...")
        
        all_results.append({
            "id": item["id"],
            "category": "Clear Non-Scam",
            "text": item["text"],
            "description": item["description"],
            "prediction": "Scam" if prediction == 1 else "Legit",
            "probability": probability,
            "confidence": confidence
        })
    
    print("\n\n")
    print("❓ TESTING AMBIGUOUS CASES")
    print("-" * 80)
    for item in ambiguous_cases:
        prediction, probability = predict_post(item["text"], return_probability=True)
        label = "🚨 SCAM" if prediction == 1 else "✅ LEGIT"
        confidence = probability if prediction == 1 else (1 - probability)
        
        print(f"\n{item['id']}: {label} (Confidence: {confidence:.1%})")
        print(f"Description: {item['description']}")
        print(f"Text: {item['text'][:150]}...")
        
        all_results.append({
            "id": item["id"],
            "category": "Ambiguous",
            "text": item["text"],
            "description": item["description"],
            "prediction": "Scam" if prediction == 1 else "Legit",
            "probability": probability,
            "confidence": confidence
        })
    
    df_results = pd.DataFrame(all_results)
    return df_results


if __name__ == "__main__":
    results = run_predictions()


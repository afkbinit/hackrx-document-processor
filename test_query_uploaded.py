import requests
import json
import time
from typing import List, Dict, Any, Tuple
import difflib
import re

# Configuration
BASE_URL = "http://localhost:8000"
HACKRX_TOKEN = "7ea833514442bb719ca788e673cbbc9fbf9555f6fa015252aaaa9679c5744303"

# Sample document URL from your test case
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Test cases with questions and expected answers
TEST_CASES = [
    {
        "question": "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "expected": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "key_terms": ["30", "thirty", "days", "grace period", "premium payment"]
    },
    {
        "question": "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "expected": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "key_terms": ["36", "thirty-six", "months", "pre-existing", "continuous coverage"]
    },
    {
        "question": "Does this policy cover maternity expenses, and what are the conditions?",
        "expected": "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "key_terms": ["yes", "maternity", "24 months", "continuously covered", "two deliveries"]
    },
    {
        "question": "What is the waiting period for cataract surgery?",
        "expected": "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "key_terms": ["2", "two", "years", "cataract", "waiting period"]
    },
    {
        "question": "Are the medical expenses for an organ donor covered under this policy?",
        "expected": "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "key_terms": ["yes", "organ donor", "medical expenses", "indemnifies", "harvesting"]
    },
    {
        "question": "What is the No Claim Discount (NCD) offered in this policy?",
        "expected": "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "key_terms": ["5%", "no claim discount", "ncd", "base premium", "renewal"]
    },
    {
        "question": "Is there a benefit for preventive health check-ups?",
        "expected": "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "key_terms": ["yes", "health check-ups", "reimburses", "two continuous policy years"]
    },
    {
        "question": "How does the policy define a 'Hospital'?",
        "expected": "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "key_terms": ["10 inpatient beds", "15 beds", "qualified nursing staff", "24/7", "operation theatre"]
    },
    {
        "question": "What is the extent of coverage for AYUSH treatments?",
        "expected": "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "key_terms": ["ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy", "ayush hospital"]
    },
    {
        "question": "Are there any sub-limits on room rent and ICU charges for Plan A?",
        "expected": "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).",
        "key_terms": ["plan a", "1%", "2%", "room rent", "icu charges", "sum insured"]
    }
]

class InsurancePolicyTester:
    def __init__(self, base_url: str = BASE_URL, token: str = HACKRX_TOKEN):
        self.base_url = base_url
        self.token = token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }
        self.test_results = []
    
    def test_health_check(self) -> bool:
        """Test if the server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def call_hackrx_api(self, questions: List[str], document_url: str = SAMPLE_DOCUMENT_URL) -> Dict[str, Any]:
        """Call the actual HackRx API and get real responses"""
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        print(f"\n🔍 Calling HackRx API with {len(questions)} questions...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=300
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get("answers", [])
                
                print(f"✅ API call successful in {processing_time:.2f} seconds")
                return {
                    "success": True,
                    "processing_time": processing_time,
                    "answers": answers,
                    "questions": questions
                }
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(f"Error: {response.text}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            print(f"❌ API call error: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_similarity_score(self, actual: str, expected: str) -> float:
        """Calculate similarity score between actual and expected answers"""
        # Normalize text
        actual_norm = re.sub(r'\s+', ' ', actual.lower().strip())
        expected_norm = re.sub(r'\s+', ' ', expected.lower().strip())
        
        # Use difflib to calculate similarity
        similarity = difflib.SequenceMatcher(None, actual_norm, expected_norm).ratio()
        return similarity
    
    def check_key_terms(self, answer: str, key_terms: List[str]) -> Tuple[float, List[str], List[str]]:
        """Check if key terms are present in the answer"""
        answer_lower = answer.lower()
        found_terms = []
        missing_terms = []
        
        for term in key_terms:
            if term.lower() in answer_lower:
                found_terms.append(term)
            else:
                missing_terms.append(term)
        
        coverage_score = len(found_terms) / len(key_terms) if key_terms else 0
        return coverage_score, found_terms, missing_terms
    
    def validate_single_answer(self, actual: str, expected: str, key_terms: List[str]) -> Dict[str, Any]:
        """Validate a single answer against expected result"""
        # Calculate text similarity
        similarity_score = self.calculate_similarity_score(actual, expected)
        
        # Check key terms coverage
        key_term_score, found_terms, missing_terms = self.check_key_terms(actual, key_terms)
        
        # Overall score (weighted average)
        overall_score = (similarity_score * 0.4) + (key_term_score * 0.6)
        
        # Determine if answer passes
        passes = overall_score >= 0.6 and key_term_score >= 0.5
        
        return {
            "similarity_score": similarity_score,
            "key_term_score": key_term_score,
            "overall_score": overall_score,
            "passes": passes,
            "found_terms": found_terms,
            "missing_terms": missing_terms
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test with actual API calls and validation"""
        print("🚀 Starting Comprehensive Insurance Policy Test")
        print("=" * 100)
        
        # 1. Health check
        if not self.test_health_check():
            return {"success": False, "error": "Health check failed"}
        
        # 2. Extract questions from test cases
        questions = [case["question"] for case in TEST_CASES]
        
        # 3. Call actual API
        api_result = self.call_hackrx_api(questions)
        
        if not api_result.get("success"):
            return {"success": False, "error": "API call failed", "details": api_result}
        
        actual_answers = api_result["answers"]
        processing_time = api_result["processing_time"]
        
        # 4. Validate each answer
        print(f"\n📊 VALIDATION RESULTS")
        print("=" * 100)
        
        test_results = []
        total_score = 0
        passed_tests = 0
        
        for i, (test_case, actual_answer) in enumerate(zip(TEST_CASES, actual_answers)):
            question = test_case["question"]
            expected_answer = test_case["expected"]
            key_terms = test_case["key_terms"]
            
            print(f"\n❓ Test {i+1}: {question}")
            print("-" * 80)
            
            # Validate answer
            validation = self.validate_single_answer(actual_answer, expected_answer, key_terms)
            
            print(f"🤖 ACTUAL ANSWER:")
            print(f"   {actual_answer}")
            print(f"\n📋 EXPECTED ANSWER:")
            print(f"   {expected_answer}")
            
            print(f"\n📈 SCORES:")
            print(f"   Text Similarity: {validation['similarity_score']:.2%}")
            print(f"   Key Terms Coverage: {validation['key_term_score']:.2%}")
            print(f"   Overall Score: {validation['overall_score']:.2%}")
            
            if validation["passes"]:
                print(f"   ✅ RESULT: PASS")
                passed_tests += 1
            else:
                print(f"   ❌ RESULT: FAIL")
            
            if validation["missing_terms"]:
                print(f"   ⚠️ Missing Key Terms: {', '.join(validation['missing_terms'])}")
            
            if validation["found_terms"]:
                print(f"   ✅ Found Key Terms: {', '.join(validation['found_terms'])}")
            
            total_score += validation['overall_score']
            
            # Store results
            test_results.append({
                "question": question,
                "actual_answer": actual_answer,
                "expected_answer": expected_answer,
                "validation": validation
            })
        
        # 5. Overall summary
        average_score = total_score / len(TEST_CASES)
        pass_rate = passed_tests / len(TEST_CASES)
        
        print(f"\n🎯 OVERALL RESULTS")
        print("=" * 100)
        print(f"📊 Tests Passed: {passed_tests}/{len(TEST_CASES)} ({pass_rate:.1%})")
        print(f"📈 Average Score: {average_score:.1%}")
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"🚀 Avg Time per Question: {processing_time/len(TEST_CASES):.2f} seconds")
        
        # Performance rating
        if pass_rate >= 0.9:
            print(f"🏆 OVERALL RATING: EXCELLENT")
        elif pass_rate >= 0.8:
            print(f"👍 OVERALL RATING: GOOD")
        elif pass_rate >= 0.6:
            print(f"⚠️ OVERALL RATING: NEEDS IMPROVEMENT")
        else:
            print(f"❌ OVERALL RATING: POOR - REQUIRES ATTENTION")
        
        return {
            "success": True,
            "passed_tests": passed_tests,
            "total_tests": len(TEST_CASES),
            "pass_rate": pass_rate,
            "average_score": average_score,
            "processing_time": processing_time,
            "test_results": test_results
        }
    
    def test_single_question_detailed(self, question_index: int) -> Dict[str, Any]:
        """Test a single question with detailed validation"""
        if question_index >= len(TEST_CASES):
            return {"success": False, "error": "Invalid question index"}
        
        test_case = TEST_CASES[question_index]
        question = test_case["question"]
        
        print(f"🔍 Testing Single Question: {question}")
        
        # Call API
        api_result = self.call_hackrx_api([question])
        
        if not api_result.get("success"):
            return {"success": False, "error": "API call failed"}
        
        actual_answer = api_result["answers"][0]
        expected_answer = test_case["expected"]
        key_terms = test_case["key_terms"]
        
        # Validate
        validation = self.validate_single_answer(actual_answer, expected_answer, key_terms)
        
        print(f"\n📋 Results:")
        print(f"Actual: {actual_answer}")
        print(f"Expected: {expected_answer}")
        print(f"Score: {validation['overall_score']:.1%}")
        print(f"Pass: {'✅ YES' if validation['passes'] else '❌ NO'}")
        
        return {
            "success": True,
            "question": question,
            "actual_answer": actual_answer,
            "expected_answer": expected_answer,
            "validation": validation
        }

def main():
    """Main test function"""
    tester = InsurancePolicyTester()
    
    print("🎯 Insurance Policy API Testing Suite")
    print("This test will:")
    print("1. Call your actual HackRx API")
    print("2. Compare real responses with expected answers")
    print("3. Validate accuracy using similarity scores and key terms")
    print("4. Provide detailed performance metrics")
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    if results.get("success"):
        # Optional: Save detailed results to file
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to test_results.json")
    
    # Optional: Test individual questions
    # single_test = tester.test_single_question_detailed(0)  # Test first question
    
    return results

if __name__ == "__main__":
    main()

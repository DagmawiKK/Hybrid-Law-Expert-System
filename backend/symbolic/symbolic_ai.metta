;; Utility function to convert numbers to MeTTa natural numbers
(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))

;; Determines validity of a contract based on various conditions
!(add-reduct &kb (: valid_contract_rule
                  (-> (Evaluation signs_contract $party1 $contract)
                      (-> (Evaluation signs_contract $party2 $contract)
                          (-> (Evaluation offers_consideration $party1 $consideration1)
                              (-> (Evaluation offers_consideration $party2 $consideration2)
                                  (-> (Evaluation accepts_offer $party1 $contract)
                                      (-> (Evaluation accepts_offer $party2 $contract)
                                          (-> (Evaluation has_capacity $party1)
                                              (-> (Evaluation has_capacity $party2)
                                                  (-> (Evaluation is_written $contract)
                                                      (-> (Evaluation meets_legal_purpose $contract)
                                                          (-> (Evaluation has_legal_intent $contract)
                                                              (Inheritance $contract valid_contract))))))))))))))

;; Determines if a contract between two parties leads to one party owing damages
!(add-reduct &kb (: owes_damages_rule
                  (-> (Evaluation breaches_contract $party $contract)
                      (-> (Inheritance $contract valid_contract)
                          (Evaluation owes_damages $party $contract)))))

;; Determines breach based on failure to perform an obligation
!(add-reduct &kb (: breach_rule
                  (-> (Evaluation has_obligation $party $obligation $contract)
                      (-> (Evaluation fails_to_perform $party $obligation $contract)
                          (Evaluation breaches_contract $party $contract)))))

;; Allows termination for material breach by part2 in contract
!(add-reduct &kb (: termination_option_rule
                    (-> (Evaluation contrat_is_between $party1 $party2 $contract)
                        (-> (Evaluation breaches_contract $party2 $contract)
                            (-> (Evaluation is_material_breach $party2 $contract)
                                (-> (Inheritance $contract valid_contract)
                                    (Evaluation can_terminate_contract $party1 $contract)))))))

;; Marks a contract as invalid if it lacks legal purpose
!(add-reduct &kb (: invalid_contract_rule
                (-> (Evaluation contrat_is_between $party1 $party2 $contract)
                    (-> (Evaluation did_not_sign $party2 $contract)
                        (Inheritance $contract invalid_contract)))))

;; Entitles parties to restitution for invalid contracts
!(add-reduct &kb (: restitution_rule
                  (-> (Inheritance $contract invalid_contract)
                      (-> (Evaluation contrat_is_between $party1 $party2 $contract)
                      (Evaluation entitled_to_restitution $party1 $contract)))))

;; Determines available remedies based on breach and validity
!(add-reduct &kb (: remedy_rule
                  (-> (Evaluation breaches_contract $party $contract)
                      (-> (Evaluation contrat_is_between $other_party $party $contract)
                            (-> (Inheritance $contract valid_contract)
                                (-> (Evaluation remedy_available $remedy)
                                    (Evaluation entitled_to_remedy $other_party $remedy $contract)))))))

;; Checks statutory compliance
!(add-reduct &kb (: statutory_compliance_rule
                  (-> (Evaluation meets_statutory_requirement $contract $requirement)
                      (Evaluation complies_with_statute $contract $requirement))))

;; Excuses performance if a condition precedent is not met
!(add-reduct &kb (: condition_precedent_rule
                  (-> (Evaluation condition_precedent $condition $party $contract)
                      (-> (Evaluation condition_not_met $condition)
                          (Evaluation excused_from_performance $party $contract)))))

;; Determines if a contract is fully performed
!(add-reduct &kb (: fully_performed_rule
                  (-> (Evaluation contrat_is_between $party1 $party2 $contract)
                      (-> (Evaluation performs_obligation $party1 $obligation1 $contract)
                          (-> (Evaluation performs_obligation $party2 $obligation2 $contract)
                              (Evaluation contract_fully_performed $contract))))))

;; Ensures enforceability based on validity and compliance
!(add-reduct &kb (: enforceability_rule
                  (-> (Inheritance $contract valid_contract)
                      (-> (Evaluation complies_with_statute $contract written_form)
                          (Evaluation enforceable_contract $contract)))))

;; Base case
(= (syn $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))

;; Recursive step
(= (syn $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* (
       ((: $prfabs (-> $prms $ccln)) (syn $kb $k (: $prfabs (-> $prms $ccln))))
          ((: $prfarg $prms) (syn $kb $k (: $prfarg $prms))))
     (: ($prfabs $prfarg) $ccln)))

;; Base case
(= (bc $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))

;; Recursive step
(= (bc $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* (((: $prfabs (-> $prms $ccln)) (bc $kb $k (: $prfabs (-> $prms $ccln))))
          ((: $prfarg $prms) (bc $kb $k (: $prfarg $prms))))
     (: ($prfabs $prfarg) $ccln)))

;; Forward DTL Curried
(: fcc (-> $a                            ; Knowledge base space
           Nat                           ; Maximum depth
           $b                            ; Source
           $b))                          ; Conclusion

;; Base case
(= (fcc $kb $_ (: $prf $prms)) (: $prf $prms))

;; Recursive step
(= (fcc $kb (S $k) (: $prfarg $prms))
   (let (: $prfabs (-> $prms $ccln)) (bc $kb $k (: $prfabs (-> $prms $ccln)))
     (fcc $kb $k (: ($prfabs $prfarg) $ccln))))
(= (fcc $kb (S $k) (: $prfabs (-> $prms $ccln)))
    (let (: $prfarg $prms) (bc $kb $k (: $prfarg $prms))
     (fcc $kb $k (: ($prfabs $prfarg) $ccln))))
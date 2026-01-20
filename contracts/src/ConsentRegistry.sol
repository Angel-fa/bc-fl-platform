// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// Το ConsentRegistry είναι smart contract που αποθηκεύει
/// τις συγκαταθέσεις ασθενών για συγκεκριμένα datasets.
/// Δεν αποθηκεύει προσωπικά δεδομένα, μόνο hashes (bytes32).
contract ConsentRegistry {

    // Ο owner είναι ο εξουσιοδοτημένος φορέας (backend/orchestrator)
    // που επιτρέπεται να γράφει ή να αλλάζει συγκαταθέσεις.
    address public owner;

    // Πιθανές καταστάσεις συγκατάθεσης
    enum ConsentStatus {
        Unknown,   // 0 → δεν υπάρχει καταχώρηση
        Allowed,   // 1 → επιτρέπεται η χρήση
        Denied     // 2 → απαγορεύεται η χρήση
    }

    // Δομή που κρατά:
    // - την κατάσταση συγκατάθεσης
    // - πότε ενημερώθηκε τελευταία φορά
    struct ConsentRecord {
        ConsentStatus status;
        uint256 updatedAt;
    }

    // Κεντρική αποθήκευση:
    // datasetKey (hash) → patientKey (hash) → ConsentRecord
    //
    // Έτσι μπορούμε να ελέγξουμε συγκατάθεση
    // χωρίς να αποθηκεύουμε raw identifiers.
    mapping(bytes32 => mapping(bytes32 => ConsentRecord)) private consents;

    // Event που εκπέμπεται κάθε φορά που αλλάζει μια συγκατάθεση.
    // Δημιουργεί audit trail στο blockchain.
    event ConsentUpdated(
        bytes32 indexed datasetKey,
        bytes32 indexed patientKey,
        ConsentStatus status,
        uint256 updatedAt
    );

    // Modifier που περιορίζει την πρόσβαση μόνο στον owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    // Constructor:
    // Ορίζει ως owner αυτόν που έκανε deploy το contract
    constructor() {
        owner = msg.sender;
    }

    // Δυνατότητα αλλαγής owner (π.χ. αν αλλάξει orchestrator)
    function setOwner(address newOwner) external onlyOwner {
        require(newOwner != address(0), "zero owner");
        owner = newOwner;
    }

    /// Καταχώρηση ή ενημέρωση συγκατάθεσης
    /// allowed = true  → Allowed
    /// allowed = false → Denied
    function setConsent(
        bytes32 datasetKey,
        bytes32 patientKey,
        bool allowed
    ) external onlyOwner {

        // Μετατροπή boolean σε enum
        ConsentStatus st = allowed
            ? ConsentStatus.Allowed
            : ConsentStatus.Denied;

        // Αποθήκευση συγκατάθεσης μαζί με timestamp
        consents[datasetKey][patientKey] = ConsentRecord({
            status: st,
            updatedAt: block.timestamp
        });

        // Εκπομπή event για audit / transparency
        emit ConsentUpdated(
            datasetKey,
            patientKey,
            st,
            block.timestamp
        );
    }

    // Ανάγνωση πλήρους πληροφορίας συγκατάθεσης
    function getConsent(
        bytes32 datasetKey,
        bytes32 patientKey
    )
        external
        view
        returns (ConsentStatus status, uint256 updatedAt)
    {
        ConsentRecord memory r = consents[datasetKey][patientKey];
        return (r.status, r.updatedAt);
    }

    // Απλός έλεγχος: υπάρχει ενεργή συγκατάθεση;
    // Επιστρέφει true μόνο αν status == Allowed
    function hasConsent(
        bytes32 datasetKey,
        bytes32 patientKey
    )
        external
        view
        returns (bool)
    {
        ConsentRecord memory r = consents[datasetKey][patientKey];
        return r.status == ConsentStatus.Allowed;
    }
}

"""

Το ConsentRegistry είναι το smart contract που χρησιμοποιεί η πλατφόρμα για να αποθηκεύει αν ένας ασθενής έχει δώσει
ή όχι συγκατάθεση για τη χρήση των δεδομένων του σε ένα συγκεκριμένο dataset. Οι συγκαταθέσεις δεν αποθηκεύονται με προσωπικά στοιχεία,
αλλά με κατακερματισμένα κλειδιά (hashes), ώστε να διασφαλίζεται η ιδιωτικότητα.
Μόνο το backend της πλατφόρμας έχει δικαίωμα να καταχωρεί ή να αλλάζει συγκαταθέσεις,
ενώ κάθε αλλαγή καταγράφεται στο blockchain μέσω events, δημιουργώντας ένα διαφανές και αμετάβλητο audit trail.
Έτσι, η πλατφόρμα μπορεί να ελέγχει με ασφάλεια αν επιτρέπεται η χρήση δεδομένων, χωρίς να μεταφέρονται
ή να αποκαλύπτονται ευαίσθητες πληροφορίες.

---
Το ConsentRegistry λέει αν επιτρέπεται η χρήση δεδομένων,
ενώ το ReceiptRegistry αποδεικνύει ότι μια ενέργεια συνέβη, χωρίς να αποκαλύπτει δεδομένα
"""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// Το ReceiptRegistry είναι smart contract που χρησιμοποιείται
/// για να "αγκυρώνει" (anchor) γεγονότα της πλατφόρμας στο blockchain.
/// Δεν αποθηκεύει δεδομένα, αλλά αποτυπώματα (hashes) για audit trail.
contract ReceiptRegistry {

    // Ο owner είναι το backend/orchestrator της πλατφόρμας
    // Μόνο αυτός επιτρέπεται να καταχωρεί νέα receipts.
    address public owner;

    // Δομή που περιγράφει ένα receipt (απόδειξη γεγονότος)
    struct Receipt {
        // Τύπος γεγονότος (π.χ. DATASET_CREATED, CONSENT_SUBMITTED)
        string eventType;

        // Αναγνωριστικό αναφοράς (π.χ. dataset_id, request_id)
        string refId;

        // Hash του payload (τι δεδομένα χρησιμοποιήθηκαν)
        bytes32 payloadHash;

        // Hash του actor (ποιος ενήργησε – χρήστης/ρόλος)
        bytes32 actorHash;

        // Χρονική στιγμή καταγραφής στο blockchain
        uint256 timestamp;
    }

    // Αποθήκευση receipts:
    // receiptId (hash) → Receipt
    mapping(bytes32 => Receipt) private receipts;

    // Event που εκπέμπεται κάθε φορά που "αγκυρώνεται" ένα receipt
    // Χρησιμοποιείται για audit trail και εξωτερική παρακολούθηση
    event ReceiptAnchored(
        bytes32 indexed receiptId,
        string eventType,
        string refId,
        bytes32 payloadHash,
        bytes32 actorHash,
        uint256 timestamp
    );

    // Modifier: επιτρέπει την εκτέλεση μόνο στον owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    // Constructor:
    // Ο owner ορίζεται αυτόματα σε όποιον κάνει deploy το contract
    constructor() {
        owner = msg.sender;
    }

    /// Καταχώρηση ενός receipt στο blockchain
    /// Δεν αποθηκεύουμε raw δεδομένα, μόνο hashes
    function anchorReceipt(
        string calldata eventType,
        string calldata refId,
        bytes32 payloadHash,
        bytes32 actorHash
    )
        external
        onlyOwner
        returns (bytes32 receiptId)
    {
        // Δημιουργία μοναδικού receiptId
        // Συνδυάζει περιεχόμενο + timestamp
        receiptId = keccak256(
            abi.encodePacked(
                eventType,
                refId,
                payloadHash,
                actorHash,
                block.timestamp
            )
        );

        // Αποθήκευση receipt στο mapping
        receipts[receiptId] = Receipt({
            eventType: eventType,
            refId: refId,
            payloadHash: payloadHash,
            actorHash: actorHash,
            timestamp: block.timestamp
        });

        // Εκπομπή event για διαφάνεια & audit
        emit ReceiptAnchored(
            receiptId,
            eventType,
            refId,
            payloadHash,
            actorHash,
            block.timestamp
        );
    }

    // Ανάκτηση ενός receipt με βάση το receiptId
    function getReceipt(bytes32 receiptId)
        external
        view
        returns (Receipt memory)
    {
        return receipts[receiptId];
    }
}

"""
Το ReceiptRegistry είναι το smart contract που λειτουργεί σαν ψηφιακό ημερολόγιο γεγονότων της πλατφόρμας.
Κάθε σημαντική ενέργεια (π.χ. δημιουργία dataset, έγκριση αιτήματος, καταχώρηση συγκατάθεσης)
δεν αποθηκεύεται με τα δεδομένα της, αλλά με ένα κρυπτογραφικό αποτύπωμα (hash) στο blockchain.
Έτσι, αποδεικνύεται ότι ένα γεγονός συνέβη, πότε συνέβη και από ποιον,
χωρίς να παραβιάζεται η ιδιωτικότητα. Το contract δημιουργεί ένα αμετάβλητο audit trail,
που ενισχύει τη διαφάνεια, την εμπιστοσύνη και τη νομική τεκμηρίωση της πλατφόρμας.

--
Το ConsentRegistry λέει αν επιτρέπεται η χρήση δεδομένων,
ενώ το ReceiptRegistry αποδεικνύει ότι μια ενέργεια συνέβη, χωρίς να αποκαλύπτει δεδομένα
"""
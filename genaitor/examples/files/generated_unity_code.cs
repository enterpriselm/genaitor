```csharp
using UnityEngine;
using System.Collections.Generic;
using XR = UnityEngine.XR.Interaction.Toolkit; // Namespace alias for clarity

//VRInteractionManager.cs
public class VRInteractionManager : MonoBehaviour
{
    public static VRInteractionManager Instance { get; private set; }

    private List<GrabbableObject> currentlyGrabbedObjects = new List<GrabbableObject>();

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this);
        }
        else
        {
            Instance = this;
        }
    }

    public void RegisterGrabbedObject(GrabbableObject obj)
    {
        if (!currentlyGrabbedObjects.Contains(obj))
        {
            currentlyGrabbedObjects.Add(obj);
        }
    }

    public void UnregisterGrabbedObject(GrabbableObject obj)
    {
        currentlyGrabbedObjects.Remove(obj);
    }

    public List<GrabbableObject> GetCurrentlyGrabbedObjects()
    {
        return currentlyGrabbedObjects;
    }

    // Example: Globally disable interactions
    public void DisableAllInteractions()
    {
        // Iterate through all GrabbableObjects and disable their XR Grab Interactable component
        GrabbableObject[] allGrabbableObjects = FindObjectsOfType<GrabbableObject>();
        foreach (GrabbableObject obj in allGrabbableObjects)
        {
            if (obj.GetComponent<XR.XRGrabInteractable>() != null)
            {
                obj.GetComponent<XR.XRGrabInteractable>().enabled = false;
            }

        }
    }

    public void EnableAllInteractions()
    {
        GrabbableObject[] allGrabbableObjects = FindObjectsOfType<GrabbableObject>();
        foreach (GrabbableObject obj in allGrabbableObjects)
        {
            if (obj.GetComponent<XR.XRGrabInteractable>() != null)
            {
                obj.GetComponent<XR.XRGrabInteractable>().enabled = true;
            }
        }
    }
}

```

```csharp
using UnityEngine;
using XR = UnityEngine.XR.Interaction.Toolkit; // Namespace alias for clarity

//GrabbableObject.cs
public class GrabbableObject : MonoBehaviour
{
    [Tooltip("Haptic intensity when grabbed (0-1)")]
    [Range(0, 1)]
    public float hapticIntensity = 0.5f;

    [Tooltip("Haptic duration when grabbed (seconds)")]
    public float hapticDuration = 0.1f;

    private XR.XRGrabInteractable grabInteractable;
    private Rigidbody rb;
    private bool isGrabbed = false;

    void Start()
    {
        grabInteractable = GetComponent<XR.XRGrabInteractable>();
        if (grabInteractable == null)
        {
            Debug.LogError("XRGrabInteractable component not found on " + gameObject.name);
            enabled = false; // Disable the script if XRGrabInteractable is missing
            return;
        }

        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError("Rigidbody component not found on " + gameObject.name);
            enabled = false;
            return;
        }

        // Subscribe to grab and release events
        grabInteractable.selectEntered.AddListener(OnGrab);
        grabInteractable.selectExited.AddListener(OnRelease);
    }

    private void OnGrab(XR.SelectEnterEventArgs args)
    {
        isGrabbed = true;
        Debug.Log("Object grabbed: " + gameObject.name);

        // Haptic feedback on the controller
        if (args.interactorObject != null && args.interactorObject.transform != null)
        {
            VRControllerInteractor controllerInteractor = args.interactorObject.transform.GetComponentInParent<VRControllerInteractor>();
            if (controllerInteractor != null)
            {
                OVRInput.Controller ovrController = controllerInteractor.GetOVRController();
                OVRInput.SetControllerVibration(hapticIntensity, hapticDuration, ovrController);
            }
            else
            {
                Debug.LogWarning("VRControllerInteractor not found on the interactor.");
            }

        }
        else
        {
            Debug.LogWarning("Interactor or its transform is null.");
        }


        // Register with the interaction manager
        VRInteractionManager.Instance.RegisterGrabbedObject(this);

        // Optional: Disable gravity while grabbed for easier manipulation
        rb.useGravity = false;
    }

    private void OnRelease(XR.SelectExitEventArgs args)
    {
        isGrabbed = false;
        Debug.Log("Object released: " + gameObject.name);

        //Haptic feedback on release (optional)
        //OVRInput.SetControllerVibration(hapticIntensity * 0.2f, hapticDuration * 0.1f, args.interactorObject.transform.GetComponentInParent<VRControllerInteractor>().GetOVRController());

        // Unregister from the interaction manager
        VRInteractionManager.Instance.UnregisterGrabbedObject(this);

        // Restore gravity
        rb.useGravity = true;
    }

    public bool IsGrabbed()
    {
        return isGrabbed;
    }
}
```

```csharp
using UnityEngine;
using XR = UnityEngine.XR.Interaction.Toolkit; // Namespace alias for clarity

//VRControllerInteractor.cs
public class VRControllerInteractor : MonoBehaviour
{
    public OVRInput.Controller controller; // Assign in the inspector.

    private XR.XRDirectInteractor directInteractor;

    void Start()
    {
        directInteractor = GetComponent<XR.XRDirectInteractor>();
        if (directInteractor == null)
        {
            Debug.LogError("XRDirectInteractor component not found on " + gameObject.name);
            enabled = false;
            return;
        }

        //Subscribe to events (optional, for more advanced interaction)
        //directInteractor.onHoverEntered.AddListener(OnHoverEntered);
        //directInteractor.onHoverExited.AddListener(OnHoverExited);
    }

    public OVRInput.Controller GetOVRController()
    {
        return controller;
    }

    // Example: Handle hover events (optional)
    //private void OnHoverEntered(XR.InteractableObject obj)
    //{
    //    // Highlight the object
    //    HighlightEffect highlight = obj.GetComponent<HighlightEffect>();
    //    if (highlight != null)
    //    {
    //        highlight.Highlight();
    //    }
    //}

    //private void OnHoverExited(XR.InteractableObject obj)
    //{
    //    // Remove highlight
    //    HighlightEffect highlight = obj.GetComponent<HighlightEffect>();
    //    if (highlight != null)
    //    {
    //        highlight.UnHighlight();
    //    }
    //}
}
```

```csharp
using UnityEngine;

//HighlightEffect.cs
public class HighlightEffect : MonoBehaviour
{
    private Renderer objectRenderer;
    private Color originalColor;
    public Color highlightColor = Color.yellow;

    void Start()
    {
        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer == null)
        {
            Debug.LogError("Renderer component not found on " + gameObject.name);
            enabled = false;
            return;
        }
        originalColor = objectRenderer.material.color;
    }

    public void Highlight()
    {
        objectRenderer.material.color = highlightColor;
    }

    public void UnHighlight()
    {
        objectRenderer.material.color = originalColor;
    }
}
```
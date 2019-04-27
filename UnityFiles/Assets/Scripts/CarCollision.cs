using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class CarCollision : MonoBehaviour
{
    public Agent agent;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
        private void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.CompareTag("obstacles"))
        {
            agent.SetReward(-50);
            // Debug.Log("Collided");
            // Debug.Log(-50);
            agent.Done();
        }
    }
}

"""
Configuration and utility functions for citation prediction methods.
Allows easy switching between old and new prediction approaches.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CitationPredictionConfig:
    """Configuration for citation prediction methods."""
    
    # Available prediction methods
    METHODS = {
        'legacy': 'Log-normal distribution (original method)',
        'improved': 'Evidence-based ensemble models',
        'auto': 'Use improved if available, fallback to legacy'
    }
    
    def __init__(self, method: str = 'auto'):
        """
        Initialize citation prediction configuration.
        
        Args:
            method: One of 'legacy', 'improved', or 'auto'
        """
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {list(self.METHODS.keys())}")
        
        self.method = method
        self.logger = logging.getLogger("citation_prediction")
    
    def get_citation_prediction(self, work: Dict[str, Any], author_stats: Optional[Dict] = None) -> float:
        """
        Get citation prediction using the configured method.
        
        Args:
            work: OpenAlex work dictionary
            author_stats: Optional precomputed author statistics
            
        Returns:
            Predicted citation count
        """
        if self.method == 'legacy':
            return self._get_legacy_prediction(work)
        elif self.method == 'improved':
            return self._get_improved_prediction(work, author_stats)
        elif self.method == 'auto':
            return self._get_auto_prediction(work, author_stats)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_legacy_prediction(self, work: Dict[str, Any]) -> float:
        """Get prediction using legacy log-normal method."""
        from .scoring import projected_citations_for_work
        
        # Force use of legacy method by temporarily removing citation_models import
        actual_citations = work.get("cited_by_count")
        if isinstance(actual_citations, int) and actual_citations > 0:
            return float(actual_citations)
        
        # Use legacy log-normal method directly
        from .scoring import log_normal_projected_citations, publication_datetime_from_work
        import datetime as dt
        
        host_venue = work.get("host_venue") or {}
        venue_stats = host_venue.get("summary_stats", {})
        impact_factor = venue_stats.get("2yr_mean_citedness", 1.0)
        
        pub_date = publication_datetime_from_work(work)
        try:
            UTC = dt.UTC
        except AttributeError:
            from datetime import timezone as _tz
            UTC = _tz.utc
        now = dt.datetime.now(UTC)
        years_since = max(0.1, (now - pub_date).days / 365.25)
        
        return log_normal_projected_citations(impact_factor, years_since)
    
    def _get_improved_prediction(self, work: Dict[str, Any], author_stats: Optional[Dict] = None) -> float:
        """Get prediction using improved evidence-based methods."""
        try:
            from .citation_models import get_improved_citation_prediction
            
            result = get_improved_citation_prediction(work, author_stats)
            
            if result.get('prediction_type') == 'actual':
                return result.get('actual_citations', 0.0)
            elif result.get('prediction_type') == 'model_based':
                return result.get('ensemble_prediction', 0.0)
            else:
                self.logger.warning("Unexpected prediction type, falling back to legacy")
                return self._get_legacy_prediction(work)
                
        except ImportError:
            self.logger.warning("Citation models not available, falling back to legacy")
            return self._get_legacy_prediction(work)
        except Exception as e:
            self.logger.warning(f"Improved prediction failed: {e}, falling back to legacy")
            return self._get_legacy_prediction(work)
    
    def _get_auto_prediction(self, work: Dict[str, Any], author_stats: Optional[Dict] = None) -> float:
        """Get prediction using auto-selection (improved with legacy fallback)."""
        try:
            return self._get_improved_prediction(work, author_stats)
        except Exception as e:
            self.logger.debug(f"Auto-selection falling back to legacy: {e}")
            return self._get_legacy_prediction(work)
    
    def get_prediction_metadata(self, work: Dict[str, Any], author_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get detailed prediction metadata including method used and confidence.
        
        Returns:
            Dictionary with prediction value, method used, confidence, and features
        """
        metadata = {
            'configured_method': self.method,
            'prediction_value': 0.0,
            'method_used': 'unknown',
            'confidence': 'unknown',
            'features': {},
            'error': None
        }
        
        try:
            if self.method == 'legacy':
                metadata['prediction_value'] = self._get_legacy_prediction(work)
                metadata['method_used'] = 'legacy'
                metadata['confidence'] = 'low'  # Due to randomness
                
            elif self.method == 'improved':
                from .citation_models import get_improved_citation_prediction
                result = get_improved_citation_prediction(work, author_stats)
                
                if result.get('prediction_type') == 'actual':
                    metadata['prediction_value'] = result.get('actual_citations', 0.0)
                    metadata['method_used'] = 'actual_citations'
                    metadata['confidence'] = 'high'
                else:
                    metadata['prediction_value'] = result.get('ensemble_prediction', 0.0)
                    metadata['method_used'] = 'ensemble'
                    metadata['confidence'] = 'medium'
                    metadata['features'] = result.get('features', {})
                    
                    # Add individual model predictions for transparency
                    metadata['model_breakdown'] = {
                        'wang_2013': result.get('wang_2013', 0.0),
                        'fu_2019': result.get('fu_2019', 0.0),
                        'sinatra_2016': result.get('sinatra_2016', 0.0)
                    }
                
            elif self.method == 'auto':
                # Try improved first
                try:
                    from .citation_models import get_improved_citation_prediction
                    result = get_improved_citation_prediction(work, author_stats)
                    
                    if result.get('prediction_type') == 'actual':
                        metadata['prediction_value'] = result.get('actual_citations', 0.0)
                        metadata['method_used'] = 'actual_citations'
                        metadata['confidence'] = 'high'
                    else:
                        metadata['prediction_value'] = result.get('ensemble_prediction', 0.0)
                        metadata['method_used'] = 'ensemble'
                        metadata['confidence'] = 'medium'
                        metadata['features'] = result.get('features', {})
                        
                except Exception as e:
                    metadata['prediction_value'] = self._get_legacy_prediction(work)
                    metadata['method_used'] = 'legacy_fallback'
                    metadata['confidence'] = 'low'
                    metadata['error'] = str(e)
                    
        except Exception as e:
            metadata['error'] = str(e)
            self.logger.error(f"Prediction failed: {e}")
            
        return metadata


# Global configuration instance (can be modified by users)
citation_config = CitationPredictionConfig(method='auto')


def set_citation_prediction_method(method: str) -> None:
    """
    Set the global citation prediction method.
    
    Args:
        method: One of 'legacy', 'improved', or 'auto'
    """
    global citation_config
    citation_config = CitationPredictionConfig(method=method)
    logger.info(f"Citation prediction method set to: {method}")


def get_citation_prediction(work: Dict[str, Any], author_stats: Optional[Dict] = None) -> float:
    """
    Get citation prediction using the globally configured method.
    
    Args:
        work: OpenAlex work dictionary
        author_stats: Optional precomputed author statistics
        
    Returns:
        Predicted citation count
    """
    return citation_config.get_citation_prediction(work, author_stats)


def get_citation_prediction_with_metadata(work: Dict[str, Any], author_stats: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get citation prediction with detailed metadata.
    
    Args:
        work: OpenAlex work dictionary
        author_stats: Optional precomputed author statistics
        
    Returns:
        Dictionary with prediction and metadata
    """
    return citation_config.get_prediction_metadata(work, author_stats)
